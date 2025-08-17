#include "core/superpoint.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

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

void SuperPointTRT::setWorkspaceSizeBytes(size_t bytes) { workspaceBytes_ = bytes; }
void SuperPointTRT::setMaxKeypoints(int maxKpts) { maxKeypoints_ = maxKpts > 0 ? maxKpts : 2048; }
void SuperPointTRT::setScoreThreshold(float thr) { scoreThreshold_ = thr; }

bool SuperPointTRT::cudaOK(cudaError_t e) {
    return e == cudaSuccess;
}

bool SuperPointTRT::init(const std::string& onnxPath, const std::string& enginePath, int height, int width) {
    if (!loadEngineFromFile(enginePath)) {
        if (!buildAndSaveEngine(onnxPath, enginePath, height, width)) return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_ || !context_->setOptimizationProfileAsync(0, nullptr)) return false;

    idx_image_ = getBindingIndexByName(*engine_, "image");
    idx_keypoints_ = getBindingIndexByName(*engine_, "keypoints");
    idx_scores_ = getBindingIndexByName(*engine_, "scores");
    idx_descriptors_ = getBindingIndexByName(*engine_, "descriptors");

    return idx_image_ >= 0 && idx_keypoints_ >= 0 && idx_scores_ >= 0 && idx_descriptors_ >= 0;
}

bool SuperPointTRT::buildAndSaveEngine(const std::string& onnxPath, const std::string& enginePath, int height, int width) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser || !parser->parseFromFile(onnxPath.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) return false;

    auto profile = builder->createOptimizationProfile();
    if (!profile) return false;

    nvinfer1::Dims dims{4, {1, 1, height, width}};
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes_);

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) return false;

    auto ser = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!ser) return false;

    std::ofstream f(enginePath, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(ser->data()), ser->size());
    return true;
}

bool SuperPointTRT::loadEngineFromFile(const std::string& enginePath) {
    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f) return false;
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> data(size);
    if (!f.read(data.data(), size)) return false;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    if (!runtime) return false;

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), size));
    return engine_ != nullptr;
}


SuperPointTRT::Result SuperPointTRT::runInference(cv::Mat& img, int height, int width) {
    img.convertTo(img, CV_32F, 1.0/255.0);
    const float* imageData = reinterpret_cast<const float*>(img.data);
    Result out;

    if (!imageData) return out;

    nvinfer1::Dims inputDims{4, {1, 1, height, width}};
    if (!context_->setInputShape("image", inputDims)) return out;

    void* d_image = nullptr; void* d_keypoints = nullptr; void* d_scores = nullptr; void* d_descriptors = nullptr;

    size_t sz_image = static_cast<size_t>(height) * static_cast<size_t>(width) * sizeof(float);
    size_t sz_keypoints = static_cast<size_t>(maxKeypoints_) * 2 * sizeof(int64_t);
    size_t sz_scores = static_cast<size_t>(maxKeypoints_) * sizeof(float);
    size_t sz_desc = static_cast<size_t>(maxKeypoints_) * 256 * sizeof(float);

    if (!cudaOK(cudaMalloc(&d_image, sz_image)) || !cudaOK(cudaMalloc(&d_keypoints, sz_keypoints)) ||
        !cudaOK(cudaMalloc(&d_scores, sz_scores)) || !cudaOK(cudaMalloc(&d_descriptors, sz_desc))) {
        cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);
        return out;
    }

    if (!cudaOK(cudaMemcpy(d_image, imageData, sz_image, cudaMemcpyHostToDevice))) {
        cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);
        return out;
    }

    const int nb = engine_->getNbIOTensors();
    std::vector<void*> bindings(nb, nullptr);
    bindings[idx_image_] = d_image;
    bindings[idx_keypoints_] = d_keypoints;
    bindings[idx_scores_] = d_scores;
    bindings[idx_descriptors_] = d_descriptors;

    if (!context_->executeV2(bindings.data())) {
        cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);
        return out;
    }

    out.keypoints.resize(static_cast<size_t>(maxKeypoints_) * 2);
    out.scores.resize(static_cast<size_t>(maxKeypoints_));
    out.descriptors.resize(static_cast<size_t>(maxKeypoints_) * 256);

    bool ok = cudaOK(cudaMemcpy(out.keypoints.data(), d_keypoints, sz_keypoints, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.scores.data(), d_scores, sz_scores, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.descriptors.data(), d_descriptors, sz_desc, cudaMemcpyDeviceToHost));

    cudaFree(d_image); cudaFree(d_keypoints); cudaFree(d_scores); cudaFree(d_descriptors);

    if (!ok) return out;

    int valid = 0;
    for (int i = 0; i < maxKeypoints_; ++i) {
        if (out.scores[i] > scoreThreshold_) ++valid;
        else break;
    }
    out.numValid = valid;
    return out;
}
