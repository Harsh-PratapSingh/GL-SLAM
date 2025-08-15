#include "core/superpoint.h"

#include <fstream>
#include <iostream>

namespace {
inline int getBindingIndexByName(const nvinfer1::ICudaEngine& eng, const char* name) {
    const int n = eng.getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* nm = eng.getIOTensorName(i);
        if (nm && std::string(nm) == name) return i;
    }
    return -1;
}
}

SuperPointTRT::SuperPointTRT() {}
SuperPointTRT::~SuperPointTRT() {}

void SuperPointTRT::setWorkspaceSizeBytes(size_t bytes) {
    workspaceBytes_ = bytes;
}

void SuperPointTRT::setMaxKeypoints(int maxKpts) {
    maxKeypoints_ = maxKpts > 0 ? maxKpts : 2048;
}

void SuperPointTRT::setScoreThreshold(float thr) {
    scoreThreshold_ = thr;
}

bool SuperPointTRT::cudaOK(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << std::endl;
        return false;
    }
    return true;
}

bool SuperPointTRT::loadEngineFromFile(const std::string& enginePath) {
    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f) return false;
    size_t size = size_t(f.tellg());
    f.seekg(0);
    std::vector<char> data(size);
    if (!f.read(data.data(), size)) return false;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    if (!runtime) return false;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), size));
    return engine_ != nullptr;
}

bool SuperPointTRT::buildAndSaveEngine(const std::string& onnxPath,
                                       const std::string& enginePath,
                                       int height, int width) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    // Use 0 flags; ONNX carries explicit batch/dynamic info
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) return false;

    if (!parser->parseFromFile(onnxPath.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX: " << onnxPath << std::endl;
        return false;
    }

    // Single-shape profile (min=opt=max) as in your current code
    auto profile = builder->createOptimizationProfile();
    if (!profile) return false;
    nvinfer1::Dims minDims{4, {1, 1, height, width}};
    nvinfer1::Dims optDims{4, {1, 1, height, width}};
    nvinfer1::Dims maxDims{4, {1, 1, height, width}};
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kMAX, maxDims);
    config->addOptimizationProfile(profile);

    // Workspace
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes_);

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) {
        std::cerr << "Failed to build engine" << std::endl;
        return false;
    }

    // Save engine
    auto ser = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!ser) {
        std::cerr << "Engine serialization failed" << std::endl;
        return false;
    }
    std::ofstream f(enginePath, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open engine file for writing: " << enginePath << std::endl;
        return false;
    }
    f.write(reinterpret_cast<const char*>(ser->data()), ser->size());
    return true;
}

bool SuperPointTRT::init(const std::string& onnxPath,
                         const std::string& enginePath,
                         int height, int width) {
    // Try to load engine; if not present build and save
    if (!loadEngineFromFile(enginePath)) {
        if (!buildAndSaveEngine(onnxPath, enginePath, height, width)) return false;
    }

    // Create context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // If multiple profiles existed, select one here (0)
    if (!context_->setOptimizationProfileAsync(0, nullptr)) {
        std::cerr << "Failed to set optimization profile 0" << std::endl;
        return false;
    }

    // Resolve binding indices by name (robust to ordering)
    idx_image_       = getBindingIndexByName(*engine_, "image");
    idx_keypoints_   = getBindingIndexByName(*engine_, "keypoints");
    idx_scores_      = getBindingIndexByName(*engine_, "scores");
    idx_descriptors_ = getBindingIndexByName(*engine_, "descriptors");

    if (idx_image_ < 0 || idx_keypoints_ < 0 || idx_scores_ < 0 || idx_descriptors_ < 0) {
        std::cerr << "Failed to resolve one or more I/O tensor indices" << std::endl;
        return false;
    }

    return true;
}

bool SuperPointTRT::runInference(const float* imageData, int height, int width, Result& out) {
    if (!imageData) {
        std::cerr << "imageData is null" << std::endl;
        return false;
    }

    // Set input shape [1,1,H,W]
    nvinfer1::Dims inputDims{4, {1, 1, height, width}};
    if (!context_->setInputShape("image", inputDims)) {
        std::cerr << "Failed to set input shape" << std::endl;
        return false;
    }

    // Device buffers
    void* d_image = nullptr;
    void* d_keypoints = nullptr;
    void* d_scores = nullptr;
    void* d_descriptors = nullptr;

    const size_t sz_image = size_t(height) * size_t(width) * sizeof(float);
    const size_t sz_keypoints = size_t(maxKeypoints_) * 2 * sizeof(int64_t);
    const size_t sz_scores    = size_t(maxKeypoints_) * sizeof(float);
    const size_t sz_desc      = size_t(maxKeypoints_) * 256 * sizeof(float);

    if (!cudaOK(cudaMalloc(&d_image, sz_image), "cudaMalloc d_image") ||
        !cudaOK(cudaMalloc(&d_keypoints, sz_keypoints), "cudaMalloc d_keypoints") ||
        !cudaOK(cudaMalloc(&d_scores, sz_scores), "cudaMalloc d_scores") ||
        !cudaOK(cudaMalloc(&d_descriptors, sz_desc), "cudaMalloc d_descriptors")) {
        if (d_image) cudaFree(d_image);
        if (d_keypoints) cudaFree(d_keypoints);
        if (d_scores) cudaFree(d_scores);
        if (d_descriptors) cudaFree(d_descriptors);
        return false;
    }

    // H2D
    if (!cudaOK(cudaMemcpy(d_image, imageData, sz_image, cudaMemcpyHostToDevice), "H2D image")) {
        cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);
        return false;
    }

    // Bindings vector using indices
    const int nb = engine_->getNbIOTensors();
    std::vector<void*> bindings(nb, nullptr);
    bindings[idx_image_]       = d_image;
    bindings[idx_keypoints_]   = d_keypoints;
    bindings[idx_scores_]      = d_scores;
    bindings[idx_descriptors_] = d_descriptors;

    // Execute
    if (!context_->executeV2(bindings.data())) {
        std::cerr << "Inference failed" << std::endl;
        cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);
        return false;
    }

    // Resize host outputs to max capacity and copy back
    out.keypoints.resize(size_t(maxKeypoints_) * 2);
    out.scores.resize(size_t(maxKeypoints_));
    out.descriptors.resize(size_t(maxKeypoints_) * 256);

    bool ok = cudaOK(cudaMemcpy(out.keypoints.data(), d_keypoints, sz_keypoints, cudaMemcpyDeviceToHost), "D2H keypoints") &&
              cudaOK(cudaMemcpy(out.scores.data(),    d_scores,    sz_scores,    cudaMemcpyDeviceToHost), "D2H scores") &&
              cudaOK(cudaMemcpy(out.descriptors.data(), d_descriptors, sz_desc,  cudaMemcpyDeviceToHost), "D2H descriptors");

    // Free device buffers
    cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);

    if (!ok) return false;

    // Count valid keypoints by threshold (assuming sorted or padded with non-positive at tail)
    int valid = 0;
    for (int i = 0; i < maxKeypoints_; ++i) {
        if (out.scores[i] > scoreThreshold_) ++valid;
        else break;
    }
    out.numValid = valid;
    return true;
}
