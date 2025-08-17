#include "core/lightglue.h"
#include <fstream>
#include <iostream>
#include <memory>


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

void LightGlueTRT::setWorkspaceSizeBytes(size_t bytes) { workspaceBytes_ = bytes; }

bool LightGlueTRT::cudaOK(cudaError_t e) {
    return e == cudaSuccess;
}

bool LightGlueTRT::init(const std::string& onnxPath, const std::string& enginePath) {
    if (!loadEngineFromFile(enginePath)) {
        if (!buildAndSaveEngine(onnxPath, enginePath)) return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_ || !context_->setOptimizationProfileAsync(0, nullptr)) return false;

    idx_kpts0_ = getBindingIndexByName(*engine_, "kpts0");
    idx_kpts1_ = getBindingIndexByName(*engine_, "kpts1");
    idx_desc0_ = getBindingIndexByName(*engine_, "desc0");
    idx_desc1_ = getBindingIndexByName(*engine_, "desc1");
    idx_matches0_ = getBindingIndexByName(*engine_, "matches0");
    idx_matches1_ = getBindingIndexByName(*engine_, "matches1");
    idx_mscores0_ = getBindingIndexByName(*engine_, "mscores0");
    idx_mscores1_ = getBindingIndexByName(*engine_, "mscores1");

    return idx_kpts0_ >= 0 && idx_kpts1_ >= 0 && idx_desc0_ >= 0 && idx_desc1_ >= 0 &&
           idx_matches0_ >= 0 && idx_matches1_ >= 0 && idx_mscores0_ >= 0 && idx_mscores1_ >= 0;
}

bool LightGlueTRT::buildAndSaveEngine(const std::string& onnxPath, const std::string& enginePath) {
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

    nvinfer1::Dims kptsDims{3, {1, 2048, 2}};
    nvinfer1::Dims descDims{3, {1, 2048, 256}};

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

bool LightGlueTRT::loadEngineFromFile(const std::string& enginePath) {
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

bool LightGlueTRT::setInputShapes(int N0, int N1) {
    if (N0 < 1 || N1 < 1) return false;

    nvinfer1::Dims kpts0Dims{3, {1, N0, 2}};
    nvinfer1::Dims kpts1Dims{3, {1, N1, 2}};
    nvinfer1::Dims desc0Dims{3, {1, N0, 256}};
    nvinfer1::Dims desc1Dims{3, {1, N1, 256}};

    return context_->setInputShape("kpts0", kpts0Dims) &&
           context_->setInputShape("kpts1", kpts1Dims) &&
           context_->setInputShape("desc0", desc0Dims) &&
           context_->setInputShape("desc1", desc1Dims);
}

bool LightGlueTRT::runInference(const std::vector<float>& kpts0, const std::vector<float>& desc0,
                                const std::vector<float>& kpts1, const std::vector<float>& desc1,
                                int N0, int N1, Result& out) {
    if (kpts0.size() != static_cast<size_t>(N0 * 2) || kpts1.size() != static_cast<size_t>(N1 * 2) ||
        desc0.size() != static_cast<size_t>(N0 * 256) || desc1.size() != static_cast<size_t>(N1 * 256)) {
        return false;
    }

    if (!setInputShapes(N0, N1)) return false;

    void* d_kpts0 = nullptr; void* d_kpts1 = nullptr; void* d_desc0 = nullptr; void* d_desc1 = nullptr;
    void* d_matches0 = nullptr; void* d_matches1 = nullptr; void* d_mscores0 = nullptr; void* d_mscores1 = nullptr;

    size_t sz_kpts0 = static_cast<size_t>(N0) * 2 * sizeof(float);
    size_t sz_kpts1 = static_cast<size_t>(N1) * 2 * sizeof(float);
    size_t sz_desc0 = static_cast<size_t>(N0) * 256 * sizeof(float);
    size_t sz_desc1 = static_cast<size_t>(N1) * 256 * sizeof(float);
    size_t sz_matches0 = static_cast<size_t>(N0) * sizeof(int64_t);
    size_t sz_matches1 = static_cast<size_t>(N1) * sizeof(int64_t);
    size_t sz_mscores0 = static_cast<size_t>(N0) * sizeof(float);
    size_t sz_mscores1 = static_cast<size_t>(N1) * sizeof(float);

    if (!cudaOK(cudaMalloc(&d_kpts0, sz_kpts0)) || !cudaOK(cudaMalloc(&d_kpts1, sz_kpts1)) ||
        !cudaOK(cudaMalloc(&d_desc0, sz_desc0)) || !cudaOK(cudaMalloc(&d_desc1, sz_desc1)) ||
        !cudaOK(cudaMalloc(&d_matches0, sz_matches0)) || !cudaOK(cudaMalloc(&d_matches1, sz_matches1)) ||
        !cudaOK(cudaMalloc(&d_mscores0, sz_mscores0)) || !cudaOK(cudaMalloc(&d_mscores1, sz_mscores1))) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return false;
    }

    if (!cudaOK(cudaMemcpy(d_kpts0, kpts0.data(), sz_kpts0, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_kpts1, kpts1.data(), sz_kpts1, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_desc0, desc0.data(), sz_desc0, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_desc1, desc1.data(), sz_desc1, cudaMemcpyHostToDevice))) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return false;
    }

    const int nb = engine_->getNbIOTensors();
    std::vector<void*> bindings(nb, nullptr);
    bindings[idx_kpts0_] = d_kpts0; bindings[idx_kpts1_] = d_kpts1;
    bindings[idx_desc0_] = d_desc0; bindings[idx_desc1_] = d_desc1;
    bindings[idx_matches0_] = d_matches0; bindings[idx_matches1_] = d_matches1;
    bindings[idx_mscores0_] = d_mscores0; bindings[idx_mscores1_] = d_mscores1;

    if (!context_->executeV2(bindings.data())) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return false;
    }

    out.matches0.resize(N0); out.matches1.resize(N1);
    out.mscores0.resize(N0); out.mscores1.resize(N1);

    bool ok = cudaOK(cudaMemcpy(out.matches0.data(), d_matches0, sz_matches0, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.matches1.data(), d_matches1, sz_matches1, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.mscores0.data(), d_mscores0, sz_mscores0, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.mscores1.data(), d_mscores1, sz_mscores1, cudaMemcpyDeviceToHost));

    cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
    cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);

    return ok;
}

LightGlueTRT::Result LightGlueTRT::run_Direct_Inference(SuperPointTRT::Result& spRes0, SuperPointTRT::Result& spRes1){

    Result out;
    const int maxKpts = 2048;
    const int spH = 376;
    const int spW = 1241;
    std::vector<float> kpts0, kpts1;
    std::vector<float> desc0, desc1;
    const int N0 = std::min(spRes0.numValid, maxKpts);
    const int N1 = std::min(spRes1.numValid, maxKpts);
    LightGlueTRT::toFloatKpts(spRes0.keypoints, N0, kpts0, spW, spH);
    LightGlueTRT::toFloatKpts(spRes1.keypoints, N1, kpts1, spW, spH);
    LightGlueTRT::sliceDescriptors(spRes0.descriptors, N0, desc0);
    LightGlueTRT::sliceDescriptors(spRes1.descriptors, N1, desc1);


    if (kpts0.size() != static_cast<size_t>(N0 * 2) || kpts1.size() != static_cast<size_t>(N1 * 2) ||
        desc0.size() != static_cast<size_t>(N0 * 256) || desc1.size() != static_cast<size_t>(N1 * 256)) {
        return out;
    }

    if (!setInputShapes(N0, N1)) return out;

    void* d_kpts0 = nullptr; void* d_kpts1 = nullptr; void* d_desc0 = nullptr; void* d_desc1 = nullptr;
    void* d_matches0 = nullptr; void* d_matches1 = nullptr; void* d_mscores0 = nullptr; void* d_mscores1 = nullptr;

    size_t sz_kpts0 = static_cast<size_t>(N0) * 2 * sizeof(float);
    size_t sz_kpts1 = static_cast<size_t>(N1) * 2 * sizeof(float);
    size_t sz_desc0 = static_cast<size_t>(N0) * 256 * sizeof(float);
    size_t sz_desc1 = static_cast<size_t>(N1) * 256 * sizeof(float);
    size_t sz_matches0 = static_cast<size_t>(N0) * sizeof(int64_t);
    size_t sz_matches1 = static_cast<size_t>(N1) * sizeof(int64_t);
    size_t sz_mscores0 = static_cast<size_t>(N0) * sizeof(float);
    size_t sz_mscores1 = static_cast<size_t>(N1) * sizeof(float);

    if (!cudaOK(cudaMalloc(&d_kpts0, sz_kpts0)) || !cudaOK(cudaMalloc(&d_kpts1, sz_kpts1)) ||
        !cudaOK(cudaMalloc(&d_desc0, sz_desc0)) || !cudaOK(cudaMalloc(&d_desc1, sz_desc1)) ||
        !cudaOK(cudaMalloc(&d_matches0, sz_matches0)) || !cudaOK(cudaMalloc(&d_matches1, sz_matches1)) ||
        !cudaOK(cudaMalloc(&d_mscores0, sz_mscores0)) || !cudaOK(cudaMalloc(&d_mscores1, sz_mscores1))) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return out;
    }

    if (!cudaOK(cudaMemcpy(d_kpts0, kpts0.data(), sz_kpts0, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_kpts1, kpts1.data(), sz_kpts1, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_desc0, desc0.data(), sz_desc0, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_desc1, desc1.data(), sz_desc1, cudaMemcpyHostToDevice))) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return out;
    }

    const int nb = engine_->getNbIOTensors();
    std::vector<void*> bindings(nb, nullptr);
    bindings[idx_kpts0_] = d_kpts0; bindings[idx_kpts1_] = d_kpts1;
    bindings[idx_desc0_] = d_desc0; bindings[idx_desc1_] = d_desc1;
    bindings[idx_matches0_] = d_matches0; bindings[idx_matches1_] = d_matches1;
    bindings[idx_mscores0_] = d_mscores0; bindings[idx_mscores1_] = d_mscores1;

    if (!context_->executeV2(bindings.data())) {
        cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
        cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);
        return out;
    }

    out.matches0.resize(N0); out.matches1.resize(N1);
    out.mscores0.resize(N0); out.mscores1.resize(N1);

    bool ok = cudaOK(cudaMemcpy(out.matches0.data(), d_matches0, sz_matches0, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.matches1.data(), d_matches1, sz_matches1, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.mscores0.data(), d_mscores0, sz_mscores0, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.mscores1.data(), d_mscores1, sz_mscores1, cudaMemcpyDeviceToHost));

    cudaFree(d_kpts0); cudaFree(d_kpts1); cudaFree(d_desc0); cudaFree(d_desc1);
    cudaFree(d_matches0); cudaFree(d_matches1); cudaFree(d_mscores0); cudaFree(d_mscores1);

    return out;

}

void LightGlueTRT::toFloatKpts(const std::vector<int64_t>& kptsIntXY, int N, std::vector<float>& kptsFloatXY, int imgWidth, int imgHeight) {
    kptsFloatXY.resize(size_t(N) * 2);
    for (int i = 0; i < N; ++i) {
        float x = static_cast<float>(kptsIntXY[size_t(i)*2 + 0]);
        float y = static_cast<float>(kptsIntXY[size_t(i)*2 + 1]);
        kptsFloatXY[size_t(i)*2 + 0] = (2.0f * x / imgWidth) - 1.0f;
        kptsFloatXY[size_t(i)*2 + 1] = (2.0f * y / imgHeight) - 1.0f;
    }
}

void LightGlueTRT::sliceDescriptors(const std::vector<float>& descAll, int N, std::vector<float>& descOut) {
    descOut.assign(descAll.begin(), descAll.begin() + size_t(N) * 256);
}


