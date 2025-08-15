#include "core/lightglue.h"

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

LightGlueTRT::LightGlueTRT() {}
LightGlueTRT::~LightGlueTRT() {}

void LightGlueTRT::setWorkspaceSizeBytes(size_t bytes) {
    workspaceBytes_ = bytes;
}

bool LightGlueTRT::cudaOK(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << std::endl;
        return false;
    }
    return true;
}

bool LightGlueTRT::init(const std::string& onnxPath, const std::string& enginePath) {
    // Try to load engine
    if (!loadEngineFromFile(enginePath)) {
        // Build and save
        if (!buildAndSaveEngine(onnxPath, enginePath)) return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // If you have multiple profiles, select one here; we use 0.
    if (!context_->setOptimizationProfileAsync(0, nullptr)) {
        std::cerr << "Failed to set optimization profile 0" << std::endl;
        return false;
    }

    // Resolve binding indices by name (robust across parser/engine ordering)
    idx_kpts0_   = getBindingIndexByName(*engine_, "kpts0");
    idx_kpts1_   = getBindingIndexByName(*engine_, "kpts1");
    idx_desc0_   = getBindingIndexByName(*engine_, "desc0");
    idx_desc1_   = getBindingIndexByName(*engine_, "desc1");
    idx_matches0_= getBindingIndexByName(*engine_, "matches0");
    idx_matches1_= getBindingIndexByName(*engine_, "matches1");
    idx_mscores0_= getBindingIndexByName(*engine_, "mscores0");
    idx_mscores1_= getBindingIndexByName(*engine_, "mscores1");

    if (idx_kpts0_ < 0 || idx_kpts1_ < 0 || idx_desc0_ < 0 || idx_desc1_ < 0 ||
        idx_matches0_ < 0 || idx_matches1_ < 0 || idx_mscores0_ < 0 || idx_mscores1_ < 0) {
        std::cerr << "Failed to resolve one or more I/O tensor indices" << std::endl;
        return false;
    }

    return true;
}

bool LightGlueTRT::buildAndSaveEngine(const std::string& onnxPath, const std::string& enginePath) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

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

    // Create optimization profile
    auto profile = builder->createOptimizationProfile();
    if (!profile) return false;

    // Ranges (adjust as needed)
    nvinfer1::Dims kptsMin{3, {1, 1, 2}};
    nvinfer1::Dims kptsOpt{3, {1, 1024, 2}};
    nvinfer1::Dims kptsMax{3, {1, 2048, 2}};

    nvinfer1::Dims descMin{3, {1, 1, 256}};
    nvinfer1::Dims descOpt{3, {1, 1024, 256}};
    nvinfer1::Dims descMax{3, {1, 2048, 256}};

    profile->setDimensions("kpts0", nvinfer1::OptProfileSelector::kMIN, kptsMin);
    profile->setDimensions("kpts0", nvinfer1::OptProfileSelector::kOPT, kptsOpt);
    profile->setDimensions("kpts0", nvinfer1::OptProfileSelector::kMAX, kptsMax);

    profile->setDimensions("kpts1", nvinfer1::OptProfileSelector::kMIN, kptsMin);
    profile->setDimensions("kpts1", nvinfer1::OptProfileSelector::kOPT, kptsOpt);
    profile->setDimensions("kpts1", nvinfer1::OptProfileSelector::kMAX, kptsMax);

    profile->setDimensions("desc0", nvinfer1::OptProfileSelector::kMIN, descMin);
    profile->setDimensions("desc0", nvinfer1::OptProfileSelector::kOPT, descOpt);
    profile->setDimensions("desc0", nvinfer1::OptProfileSelector::kMAX, descMax);

    profile->setDimensions("desc1", nvinfer1::OptProfileSelector::kMIN, descMin);
    profile->setDimensions("desc1", nvinfer1::OptProfileSelector::kOPT, descOpt);
    profile->setDimensions("desc1", nvinfer1::OptProfileSelector::kMAX, descMax);

    config->addOptimizationProfile(profile);

    // Workspace cap
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

bool LightGlueTRT::loadEngineFromFile(const std::string& enginePath) {
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

bool LightGlueTRT::setInputShapes(int N0, int N1) {
    // Validate basic constraints
    if (N0 < 1 || N1 < 1) {
        std::cerr << "N0/N1 must be >= 1" << std::endl;
        return false;
    }

    nvinfer1::Dims kpts0Dims{3, {1, N0, 2}};
    nvinfer1::Dims kpts1Dims{3, {1, N1, 2}};
    nvinfer1::Dims desc0Dims{3, {1, N0, 256}};
    nvinfer1::Dims desc1Dims{3, {1, N1, 256}};

    if (!context_->setInputShape("kpts0", kpts0Dims) ||
        !context_->setInputShape("kpts1", kpts1Dims) ||
        !context_->setInputShape("desc0", desc0Dims) ||
        !context_->setInputShape("desc1", desc1Dims)) {
        std::cerr << "Failed to set input shapes" << std::endl;
        return false;
    }
    return true;
}

bool LightGlueTRT::runInference(const std::vector<float>& kpts0,
                                const std::vector<float>& desc0,
                                const std::vector<float>& kpts1,
                                const std::vector<float>& desc1,
                                int N0, int N1,
                                Result& out) {
    // Basic validation of input sizes
    if (kpts0.size() != static_cast<size_t>(N0 * 2) ||
        kpts1.size() != static_cast<size_t>(N1 * 2) ||
        desc0.size() != static_cast<size_t>(N0 * 256) ||
        desc1.size() != static_cast<size_t>(N1 * 256)) {
        std::cerr << "Input vector sizes do not match N0/N1" << std::endl;
        return false;
    }

    if (!setInputShapes(N0, N1)) return false;

    // Device buffers
    void* d_kpts0 = nullptr;
    void* d_kpts1 = nullptr;
    void* d_desc0 = nullptr;
    void* d_desc1 = nullptr;
    void* d_matches0 = nullptr;
    void* d_matches1 = nullptr;
    void* d_mscores0 = nullptr;
    void* d_mscores1 = nullptr;

    const size_t sz_kpts0 = size_t(N0) * 2 * sizeof(float);
    const size_t sz_kpts1 = size_t(N1) * 2 * sizeof(float);
    const size_t sz_desc0 = size_t(N0) * 256 * sizeof(float);
    const size_t sz_desc1 = size_t(N1) * 256 * sizeof(float);
    const size_t sz_matches0 = size_t(N0) * sizeof(int64_t);
    const size_t sz_matches1 = size_t(N1) * sizeof(int64_t);
    const size_t sz_mscores0 = size_t(N0) * sizeof(float);
    const size_t sz_mscores1 = size_t(N1) * sizeof(float);

    if (!cudaOK(cudaMalloc(&d_kpts0, sz_kpts0), "cudaMalloc d_kpts0") ||
        !cudaOK(cudaMalloc(&d_kpts1, sz_kpts1), "cudaMalloc d_kpts1") ||
        !cudaOK(cudaMalloc(&d_desc0, sz_desc0), "cudaMalloc d_desc0") ||
        !cudaOK(cudaMalloc(&d_desc1, sz_desc1), "cudaMalloc d_desc1") ||
        !cudaOK(cudaMalloc(&d_matches0, sz_matches0), "cudaMalloc d_matches0") ||
        !cudaOK(cudaMalloc(&d_matches1, sz_matches1), "cudaMalloc d_matches1") ||
        !cudaOK(cudaMalloc(&d_mscores0, sz_mscores0), "cudaMalloc d_mscores0") ||
        !cudaOK(cudaMalloc(&d_mscores1, sz_mscores1), "cudaMalloc d_mscores1")) {
        // Cleanup partial allocations
        if (d_kpts0) cudaFree(d_kpts0); if (d_kpts1) cudaFree(d_kpts1);
        if (d_desc0) cudaFree(d_desc0); if (d_desc1) cudaFree(d_desc1);
        if (d_matches0) cudaFree(d_matches0); if (d_matches1) cudaFree(d_matches1);
        if (d_mscores0) cudaFree(d_mscores0); if (d_mscores1) cudaFree(d_mscores1);
        return false;
    }

    // H2D
    if (!cudaOK(cudaMemcpy(d_kpts0, kpts0.data(), sz_kpts0, cudaMemcpyHostToDevice), "H2D kpts0") ||
        !cudaOK(cudaMemcpy(d_kpts1, kpts1.data(), sz_kpts1, cudaMemcpyHostToDevice), "H2D kpts1") ||
        !cudaOK(cudaMemcpy(d_desc0, desc0.data(), sz_desc0, cudaMemcpyHostToDevice), "H2D desc0") ||
        !cudaOK(cudaMemcpy(d_desc1, desc1.data(), sz_desc1, cudaMemcpyHostToDevice), "H2D desc1")) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return false;
    }

    // Prepare bindings array matching engine binding indices
    const int nb = engine_->getNbIOTensors();
    std::vector<void*> bindings(nb, nullptr);
    bindings[idx_kpts0_]    = d_kpts0;
    bindings[idx_kpts1_]    = d_kpts1;
    bindings[idx_desc0_]    = d_desc0;
    bindings[idx_desc1_]    = d_desc1;
    bindings[idx_matches0_] = d_matches0;
    bindings[idx_matches1_] = d_matches1;
    bindings[idx_mscores0_] = d_mscores0;
    bindings[idx_mscores1_] = d_mscores1;

    // Execute
    if (!context_->executeV2(bindings.data())) {
        std::cerr << "Inference failed" << std::endl;
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return false;
    }

    // Resize outputs and D2H
    out.matches0.resize(N0);
    out.matches1.resize(N1);
    out.mscores0.resize(N0);
    out.mscores1.resize(N1);

    bool ok = cudaOK(cudaMemcpy(out.matches0.data(), d_matches0, sz_matches0, cudaMemcpyDeviceToHost), "D2H matches0") &&
              cudaOK(cudaMemcpy(out.matches1.data(), d_matches1, sz_matches1, cudaMemcpyDeviceToHost), "D2H matches1") &&
              cudaOK(cudaMemcpy(out.mscores0.data(), d_mscores0, sz_mscores0, cudaMemcpyDeviceToHost), "D2H mscores0") &&
              cudaOK(cudaMemcpy(out.mscores1.data(), d_mscores1, sz_mscores1, cudaMemcpyDeviceToHost), "D2H mscores1");

    // Cleanup
    cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
    cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);

    return ok;
}
