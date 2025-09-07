#include "core/keypt2subpx.h"
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

Keypt2SubpxTRT::Keypt2SubpxTRT() {}
Keypt2SubpxTRT::~Keypt2SubpxTRT() {}

void Keypt2SubpxTRT::setWorkspaceSizeBytes(size_t bytes) { workspaceBytes_ = bytes; }

bool Keypt2SubpxTRT::cudaOK(cudaError_t e) {
    return e == cudaSuccess;
}

bool Keypt2SubpxTRT::init(const std::string& onnxPath, const std::string& enginePath) {
    if (!loadEngineFromFile(enginePath)) {
        if (!buildAndSaveEngine(onnxPath, enginePath)) return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_ || !context_->setOptimizationProfileAsync(0, nullptr)) return false;

    idx_keypt0_ = getBindingIndexByName(*engine_, "keypt1");
    idx_keypt1_ = getBindingIndexByName(*engine_, "keypt2");
    idx_img0_ = getBindingIndexByName(*engine_, "img1");
    idx_img1_ = getBindingIndexByName(*engine_, "img2");
    idx_desc0_ = getBindingIndexByName(*engine_, "desc1");
    idx_desc1_ = getBindingIndexByName(*engine_, "desc2");
    idx_score0_ = getBindingIndexByName(*engine_, "score1");
    idx_score1_ = getBindingIndexByName(*engine_, "score2");
    idx_refined_keypt0_ = getBindingIndexByName(*engine_, "refined_keypt1");
    idx_refined_keypt1_ = getBindingIndexByName(*engine_, "refined_keypt2");

    return idx_keypt0_ >= 0 && idx_keypt1_ >= 0 && idx_img0_ >= 0 && idx_img1_ >= 0 &&
           idx_desc0_ >= 0 && idx_desc1_ >= 0 && idx_score0_ >= 0 && idx_score1_ >= 0 &&
           idx_refined_keypt0_ >= 0 && idx_refined_keypt1_ >= 0;
}

bool Keypt2SubpxTRT::buildAndSaveEngine(const std::string& onnxPath, const std::string& enginePath) {
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

    // Keypt dims (N, 2)
    nvinfer1::Dims kptMin{2, {1, 2}};
    nvinfer1::Dims kptOpt{2, {1024, 2}};
    nvinfer1::Dims kptMax{2, {2048, 2}};

    // Desc dims (N, 256)
    nvinfer1::Dims descMin{2, {1, 256}};
    nvinfer1::Dims descOpt{2, {1024, 256}};
    nvinfer1::Dims descMax{2, {2048, 256}};

    // Img dims (3, H, W)
    nvinfer1::Dims imgMin{3, {3, 128, 128}};
    nvinfer1::Dims imgOpt{3, {3, 768, 1024}};
    nvinfer1::Dims imgMax{3, {3, 1080, 1920}};

    // Score dims (N,) - adjusted to per-keypoint/match scores
    // In buildAndSaveEngine, adjust the score dimensions to 3D [1, H, W] to match ONNX

    // Score dims (1, H, W)
    nvinfer1::Dims scoreMin{3, {1, 128, 128}};
    nvinfer1::Dims scoreOpt{3, {1, 768, 1024}};
    nvinfer1::Dims scoreMax{3, {1, 1080, 1920}};

    profile->setDimensions("keypt1", nvinfer1::OptProfileSelector::kMIN, kptMin);
    profile->setDimensions("keypt1", nvinfer1::OptProfileSelector::kOPT, kptOpt);
    profile->setDimensions("keypt1", nvinfer1::OptProfileSelector::kMAX, kptMax);

    profile->setDimensions("keypt2", nvinfer1::OptProfileSelector::kMIN, kptMin);
    profile->setDimensions("keypt2", nvinfer1::OptProfileSelector::kOPT, kptOpt);
    profile->setDimensions("keypt2", nvinfer1::OptProfileSelector::kMAX, kptMax);

    profile->setDimensions("desc1", nvinfer1::OptProfileSelector::kMIN, descMin);
    profile->setDimensions("desc1", nvinfer1::OptProfileSelector::kOPT, descOpt);
    profile->setDimensions("desc1", nvinfer1::OptProfileSelector::kMAX, descMax);

    profile->setDimensions("desc2", nvinfer1::OptProfileSelector::kMIN, descMin);
    profile->setDimensions("desc2", nvinfer1::OptProfileSelector::kOPT, descOpt);
    profile->setDimensions("desc2", nvinfer1::OptProfileSelector::kMAX, descMax);

    profile->setDimensions("img1", nvinfer1::OptProfileSelector::kMIN, imgMin);
    profile->setDimensions("img1", nvinfer1::OptProfileSelector::kOPT, imgOpt);
    profile->setDimensions("img1", nvinfer1::OptProfileSelector::kMAX, imgMax);

    profile->setDimensions("img2", nvinfer1::OptProfileSelector::kMIN, imgMin);
    profile->setDimensions("img2", nvinfer1::OptProfileSelector::kOPT, imgOpt);
    profile->setDimensions("img2", nvinfer1::OptProfileSelector::kMAX, imgMax);

    profile->setDimensions("score1", nvinfer1::OptProfileSelector::kMIN, scoreMin);
    profile->setDimensions("score1", nvinfer1::OptProfileSelector::kOPT, scoreOpt);
    profile->setDimensions("score1", nvinfer1::OptProfileSelector::kMAX, scoreMax);

    profile->setDimensions("score2", nvinfer1::OptProfileSelector::kMIN, scoreMin);
    profile->setDimensions("score2", nvinfer1::OptProfileSelector::kOPT, scoreOpt);
    profile->setDimensions("score2", nvinfer1::OptProfileSelector::kMAX, scoreMax);

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

bool Keypt2SubpxTRT::loadEngineFromFile(const std::string& enginePath) {
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

bool Keypt2SubpxTRT::setInputShapes(int N, int H, int W) {
    if (N < 1 || H < 1 || W < 1) return false;

    nvinfer1::Dims keyptDims{2, {N, 2}};
    nvinfer1::Dims descDims{2, {N, 256}};
    nvinfer1::Dims imgDims{3, {3, H, W}};
    nvinfer1::Dims scoreDims{3, {1, H, W}};

    return context_->setInputShape("keypt1", keyptDims) &&
           context_->setInputShape("keypt2", keyptDims) &&
           context_->setInputShape("desc1", descDims) &&
           context_->setInputShape("desc2", descDims) &&
           context_->setInputShape("img1", imgDims) &&
           context_->setInputShape("img2", imgDims) &&
           context_->setInputShape("score1", scoreDims) &&
           context_->setInputShape("score2", scoreDims);
}

bool Keypt2SubpxTRT::runInference(const std::vector<float>& keypt0, const std::vector<float>& keypt1,
                                  const std::vector<float>& img0, const std::vector<float>& img1,
                                  const std::vector<float>& desc0, const std::vector<float>& desc1,
                                  const std::vector<float>& score0, const std::vector<float>& score1,
                                  int N, int H, int W, Result& out) {
    // Fix the size check for scores - they should be H*W, not N
    if (keypt0.size() != static_cast<size_t>(N * 2) || keypt1.size() != static_cast<size_t>(N * 2) ||
        img0.size() != static_cast<size_t>(3 * H * W) || img1.size() != static_cast<size_t>(3 * H * W) ||
        desc0.size() != static_cast<size_t>(N * 256) || desc1.size() != static_cast<size_t>(N * 256) ||
        score0.size() != static_cast<size_t>(H * W) || score1.size() != static_cast<size_t>(H * W)) {  // FIXED
        return false;
    }

    if (!setInputShapes(N, H, W)) return false;

    void* d_keypt0 = nullptr; void* d_keypt1 = nullptr; void* d_img0 = nullptr; void* d_img1 = nullptr;
    void* d_desc0 = nullptr; void* d_desc1 = nullptr; void* d_score0 = nullptr; void* d_score1 = nullptr;
    void* d_refined0 = nullptr; void* d_refined1 = nullptr;

    size_t sz_keypt = static_cast<size_t>(N) * 2 * sizeof(float);
    size_t sz_img = static_cast<size_t>(3) * H * W * sizeof(float);
    size_t sz_desc = static_cast<size_t>(N) * 256 * sizeof(float);
    size_t sz_score = static_cast<size_t>(H * W) * sizeof(float);  // FIXED: Should be H*W, not N
    size_t sz_refined = sz_keypt;

    if (!cudaOK(cudaMalloc(&d_keypt0, sz_keypt)) || !cudaOK(cudaMalloc(&d_keypt1, sz_keypt)) ||
        !cudaOK(cudaMalloc(&d_img0, sz_img)) || !cudaOK(cudaMalloc(&d_img1, sz_img)) ||
        !cudaOK(cudaMalloc(&d_desc0, sz_desc)) || !cudaOK(cudaMalloc(&d_desc1, sz_desc)) ||
        !cudaOK(cudaMalloc(&d_score0, sz_score)) || !cudaOK(cudaMalloc(&d_score1, sz_score)) ||
        !cudaOK(cudaMalloc(&d_refined0, sz_refined)) || !cudaOK(cudaMalloc(&d_refined1, sz_refined))) {
        cudaFree(d_keypt0); cudaFree(d_keypt1); cudaFree(d_img0); cudaFree(d_img1);
        cudaFree(d_desc0); cudaFree(d_desc1); cudaFree(d_score0); cudaFree(d_score1);
        cudaFree(d_refined0); cudaFree(d_refined1);
        return false;
    }

    if (!cudaOK(cudaMemcpy(d_keypt0, keypt0.data(), sz_keypt, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_keypt1, keypt1.data(), sz_keypt, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_img0, img0.data(), sz_img, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_img1, img1.data(), sz_img, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_desc0, desc0.data(), sz_desc, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_desc1, desc1.data(), sz_desc, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_score0, score0.data(), sz_score, cudaMemcpyHostToDevice)) ||
        !cudaOK(cudaMemcpy(d_score1, score1.data(), sz_score, cudaMemcpyHostToDevice))) {
        cudaFree(d_keypt0); cudaFree(d_keypt1); cudaFree(d_img0); cudaFree(d_img1);
        cudaFree(d_desc0); cudaFree(d_desc1); cudaFree(d_score0); cudaFree(d_score1);
        cudaFree(d_refined0); cudaFree(d_refined1);
        return false;
    }

    const int nb = engine_->getNbIOTensors();
    std::vector<void*> bindings(nb, nullptr);
    bindings[idx_keypt0_] = d_keypt0; bindings[idx_keypt1_] = d_keypt1;
    bindings[idx_img0_] = d_img0; bindings[idx_img1_] = d_img1;
    bindings[idx_desc0_] = d_desc0; bindings[idx_desc1_] = d_desc1;
    bindings[idx_score0_] = d_score0; bindings[idx_score1_] = d_score1;
    bindings[idx_refined_keypt0_] = d_refined0; bindings[idx_refined_keypt1_] = d_refined1;

    if (!context_->executeV2(bindings.data())) {
        cudaFree(d_keypt0); cudaFree(d_keypt1); cudaFree(d_img0); cudaFree(d_img1);
        cudaFree(d_desc0); cudaFree(d_desc1); cudaFree(d_score0); cudaFree(d_score1);
        cudaFree(d_refined0); cudaFree(d_refined1);
        return false;
    }

    out.refined_keypt0.resize(N * 2); out.refined_keypt1.resize(N * 2);

    bool ok = cudaOK(cudaMemcpy(out.refined_keypt0.data(), d_refined0, sz_refined, cudaMemcpyDeviceToHost)) &&
              cudaOK(cudaMemcpy(out.refined_keypt1.data(), d_refined1, sz_refined, cudaMemcpyDeviceToHost));

    cudaFree(d_keypt0); cudaFree(d_keypt1); cudaFree(d_img0); cudaFree(d_img1);
    cudaFree(d_desc0); cudaFree(d_desc1); cudaFree(d_score0); cudaFree(d_score1);
    cudaFree(d_refined0); cudaFree(d_refined1);

    return ok;
}

Keypt2SubpxTRT::Result Keypt2SubpxTRT::run_Direct_Inference(const LightGlueTRT::Result& lgRes,
                                                            const cv::Mat& img0,
                                                            const cv::Mat& img1) {
    Result out;

    const int maxKpts = 2048;
    const int spH = img0.rows;  // Use actual image height
    const int spW = img0.cols;  // Use actual image width

    // Convert grayscale to 3-channel BGR
    cv::Mat img0_rgb, img1_rgb;
    cv::cvtColor(img0, img0_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img1, img1_rgb, cv::COLOR_GRAY2BGR);

    // Convert to float and normalize [0,1]
    img0_rgb.convertTo(img0_rgb, CV_32FC3, 1.0 / 255.0);
    img1_rgb.convertTo(img1_rgb, CV_32FC3, 1.0 / 255.0);

    // Access contiguous data as vectors (3 * H * W)
    std::vector<float> imgVec0(reinterpret_cast<const float*>(img0_rgb.data),
                               reinterpret_cast<const float*>(img0_rgb.data) + (3 * spH * spW));
    std::vector<float> imgVec1(reinterpret_cast<const float*>(img1_rgb.data),
                               reinterpret_cast<const float*>(img1_rgb.data) + (3 * spH * spW));

    int N;
    std::vector<float> m_keypt0, m_keypt1, m_desc0, m_desc1, m_score0, m_score1;
    extractMatched(lgRes, m_keypt0, m_keypt1, m_desc0, m_desc1, m_score0, m_score1, N, spH, spW);  // Pass H, W

    // Convert keypoints to float (assuming they are int64_t in lgRes, convert to pixel float)
    // No normalization needed for Keypt2Subpx (raw pixel coords)

    runInference(m_keypt0, m_keypt1, imgVec0, imgVec1, m_desc0, m_desc1, m_score0, m_score1, N, spH, spW, out);

    return out;
}

void Keypt2SubpxTRT::extractMatched(const LightGlueTRT::Result& lgRes, std::vector<float>& m_keypt0, std::vector<float>& m_keypt1,
                                    std::vector<float>& m_desc0, std::vector<float>& m_desc1,
                                    std::vector<float>& m_score0, std::vector<float>& m_score1,
                                    int& N, int H, int W) {  // Add H, W parameters
    N = 0;
    for (size_t i = 0; i < lgRes.matches0.size(); ++i) {
        if (lgRes.matches0[i] != -1) ++N;
    }

    m_keypt0.resize(static_cast<size_t>(N) * 2);
    m_keypt1.resize(static_cast<size_t>(N) * 2);
    m_desc0.resize(static_cast<size_t>(N) * 256);
    m_desc1.resize(static_cast<size_t>(N) * 256);
    
    // FIXED: Create full score maps instead of per-keypoint scores
    m_score0.resize(static_cast<size_t>(H * W), 0.0f);  // Initialize to zeros
    m_score1.resize(static_cast<size_t>(H * W), 0.0f);  // Initialize to zeros

    int idx = 0;
    for (size_t i = 0; i < lgRes.matches0.size(); ++i) {
        int64_t j = lgRes.matches0[i];
        if (j == -1) continue;

        // keypt0 (convert int64_t to float)
        m_keypt0[idx * 2 + 0] = static_cast<float>(lgRes.keypoints0[i * 2 + 0]);
        m_keypt0[idx * 2 + 1] = static_cast<float>(lgRes.keypoints0[i * 2 + 1]);

        // keypt1
        m_keypt1[idx * 2 + 0] = static_cast<float>(lgRes.keypoints1[j * 2 + 0]);
        m_keypt1[idx * 2 + 1] = static_cast<float>(lgRes.keypoints1[j * 2 + 1]);

        // desc0
        for (int k = 0; k < 256; ++k) {
            m_desc0[idx * 256 + k] = lgRes.descriptors0[i * 256 + k];
        }

        // desc1
        for (int k = 0; k < 256; ++k) {
            m_desc1[idx * 256 + k] = lgRes.descriptors1[j * 256 + k];
        }

        // FIXED: Scatter scores into 2D maps at keypoint locations
        int x0 = static_cast<int>(lgRes.keypoints0[i * 2 + 0]);
        int y0 = static_cast<int>(lgRes.keypoints0[i * 2 + 1]);
        int x1 = static_cast<int>(lgRes.keypoints1[j * 2 + 0]);
        int y1 = static_cast<int>(lgRes.keypoints1[j * 2 + 1]);
        
        // Ensure coordinates are within bounds
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            m_score0[y0 * W + x0] = lgRes.mscores0[i];
        }
        if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
            m_score1[y1 * W + x1] = lgRes.mscores1[j];
        }

        ++idx;
    }
}

