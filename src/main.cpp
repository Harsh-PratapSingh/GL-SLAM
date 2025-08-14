#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

// Simple TensorRT logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

static bool cudaCheck(cudaError_t e, const char* m) {
    if (e != cudaSuccess) {
        std::cerr << m << ": " << cudaGetErrorString(e) << std::endl;
        return false;
    }
    return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> buildEngine(const std::string& onnxFile, Logger& logger) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) return nullptr;

    // kEXPLICIT_BATCH is deprecated in recent TensorRT versions; use 0 flags for dynamic shapes as parsed from ONNX
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) return nullptr;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return nullptr;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) return nullptr;

    if (!parser->parseFromFile(onnxFile.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX: " << onnxFile << std::endl;
        return nullptr;
    }

    // Create optimization profile for dynamic inputs
    auto profile = builder->createOptimizationProfile();
    if (!profile) return nullptr;

    // Inputs (from Netron):
    // kpts0: float32[1, num_keypoints0, 2]
    // kpts1: float32[1, num_keypoints1, 2]
    // desc0: float32[1, num_keypoints0, 256]
    // desc1: float32[1, num_keypoints1, 256]

    // Choose reasonable ranges; adjust to your app’s needs.
    // min must be >=1; opt typical usage; max up to 2048 (SuperPoint cap).
    // You can also set multiple profiles if you have a few fixed sizes.
    nvinfer1::Dims kptsMin{3, {1, 1, 2}};
    nvinfer1::Dims kptsOpt{3, {1, 1024, 2}};
    nvinfer1::Dims kptsMax{3, {1, 2048, 2}};

    nvinfer1::Dims descMin{3, {1, 1, 256}};
    nvinfer1::Dims descOpt{3, {1, 1024, 256}};
    nvinfer1::Dims descMax{3, {1, 2048, 256}};

    // For kpts0/desc0 use opt0; for kpts1/desc1 use opt1 (purely as an example).
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

    // Workspace limit
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string& engineFile, Logger& logger) {
    std::ifstream f(engineFile, std::ios::binary | std::ios::ate);
    if (!f) return nullptr;
    size_t size = size_t(f.tellg());
    f.seekg(0);
    std::vector<char> data(size);
    if (!f.read(data.data(), size)) return nullptr;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) return nullptr;
    return std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), size));
}

void saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& engineFile) {
    auto ser = std::unique_ptr<nvinfer1::IHostMemory>(engine.serialize());
    if (!ser) {
        std::cerr << "Engine serialization failed" << std::endl;
        return;
    }
    std::ofstream f(engineFile, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open engine file: " << engineFile << std::endl;
        return;
    }
    f.write(reinterpret_cast<const char*>(ser->data()), ser->size());
}

int main() {
    Logger logger;

    const std::string onnxPath = "superpoint_lightglue.onnx";
    const std::string enginePath = "superpoint_lightglue.engine";

    // Load or build engine
    auto engine = loadEngine(enginePath, logger);
    if (!engine) {
        std::cout << "Building engine from ONNX..." << std::endl;
        engine = buildEngine(onnxPath, logger);
        if (!engine) {
            std::cerr << "Failed to build engine" << std::endl;
            return -1;
        }
        saveEngine(*engine, enginePath);
        std::cout << "Engine saved to " << enginePath << std::endl;
    } else {
        std::cout << "Engine loaded from " << enginePath << std::endl;
    }

    // Create execution context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return -1;
    }

    // Select the optimization profile if multiple are present (index 0 here)
    if (!context->setOptimizationProfileAsync(0, nullptr)) {
        // If using streams, pass a valid stream; nullptr is acceptable in many cases, but if it fails, create a CUDA stream and pass it.
        std::cerr << "Failed to set optimization profile 0" << std::endl;
        return -1;
    }

    // Dummy input sizes (within profile ranges)
    const int N0 = 512;  // num_keypoints0
    const int N1 = 640;  // num_keypoints1

    // Set input shapes
    nvinfer1::Dims kpts0Dims{3, {1, N0, 2}};
    nvinfer1::Dims kpts1Dims{3, {1, N1, 2}};
    nvinfer1::Dims desc0Dims{3, {1, N0, 256}};
    nvinfer1::Dims desc1Dims{3, {1, N1, 256}};

    if (!context->setInputShape("kpts0", kpts0Dims) ||
        !context->setInputShape("kpts1", kpts1Dims) ||
        !context->setInputShape("desc0", desc0Dims) ||
        !context->setInputShape("desc1", desc1Dims)) {
        std::cerr << "Failed to set input shapes" << std::endl;
        return -1;
    }

    // Allocate and fill dummy host inputs
    std::vector<float> kpts0(1LL * N0 * 2, 0.0f);
    std::vector<float> kpts1(1LL * N1 * 2, 0.0f);
    std::vector<float> desc0(1LL * N0 * 256, 0.0f);
    std::vector<float> desc1(1LL * N1 * 256, 0.0f);

    // Simple deterministic dummy data
    for (int i = 0; i < N0; ++i) {
        kpts0[2*i+0] = float(i % 640); // x
        kpts0[2*i+1] = float(i / 640); // y
        for (int d = 0; d < 256; ++d) {
            desc0[i*256 + d] = (d % 2 == 0) ? 0.1f : -0.1f;
        }
    }
    for (int i = 0; i < N1; ++i) {
        kpts1[2*i+0] = float(i % 640);
        kpts1[2*i+1] = float(i / 640);
        for (int d = 0; d < 256; ++d) {
            desc1[i*256 + d] = (d % 3 == 0) ? 0.2f : -0.05f;
        }
    }

    // Device buffers
    void* d_kpts0 = nullptr;
    void* d_kpts1 = nullptr;
    void* d_desc0 = nullptr;
    void* d_desc1 = nullptr;
    void* d_matches0 = nullptr;
    void* d_matches1 = nullptr;
    void* d_mscores0 = nullptr;
    void* d_mscores1 = nullptr;

    size_t sz_kpts0 = size_t(1) * N0 * 2 * sizeof(float);
    size_t sz_kpts1 = size_t(1) * N1 * 2 * sizeof(float);
    size_t sz_desc0 = size_t(1) * N0 * 256 * sizeof(float);
    size_t sz_desc1 = size_t(1) * N1 * 256 * sizeof(float);
    size_t sz_matches0 = size_t(1) * N0 * sizeof(int64_t);
    size_t sz_matches1 = size_t(1) * N1 * sizeof(int64_t);
    size_t sz_mscores0 = size_t(1) * N0 * sizeof(float);
    size_t sz_mscores1 = size_t(1) * N1 * sizeof(float);

    if (!cudaCheck(cudaMalloc(&d_kpts0, sz_kpts0), "cudaMalloc d_kpts0") ||
        !cudaCheck(cudaMalloc(&d_kpts1, sz_kpts1), "cudaMalloc d_kpts1") ||
        !cudaCheck(cudaMalloc(&d_desc0, sz_desc0), "cudaMalloc d_desc0") ||
        !cudaCheck(cudaMalloc(&d_desc1, sz_desc1), "cudaMalloc d_desc1") ||
        !cudaCheck(cudaMalloc(&d_matches0, sz_matches0), "cudaMalloc d_matches0") ||
        !cudaCheck(cudaMalloc(&d_matches1, sz_matches1), "cudaMalloc d_matches1") ||
        !cudaCheck(cudaMalloc(&d_mscores0, sz_mscores0), "cudaMalloc d_mscores0") ||
        !cudaCheck(cudaMalloc(&d_mscores1, sz_mscores1), "cudaMalloc d_mscores1")) {
        return -1;
    }

    // H2D
    if (!cudaCheck(cudaMemcpy(d_kpts0, kpts0.data(), sz_kpts0, cudaMemcpyHostToDevice), "H2D kpts0") ||
        !cudaCheck(cudaMemcpy(d_kpts1, kpts1.data(), sz_kpts1, cudaMemcpyHostToDevice), "H2D kpts1") ||
        !cudaCheck(cudaMemcpy(d_desc0, desc0.data(), sz_desc0, cudaMemcpyHostToDevice), "H2D desc0") ||
        !cudaCheck(cudaMemcpy(d_desc1, desc1.data(), sz_desc1, cudaMemcpyHostToDevice), "H2D desc1")) {
        return -1;
    }

    // Bindings must match engine I/O order.
    // From Netron names: inputs [kpts0, kpts1, desc0, desc1], outputs [matches0, matches1, mscores0, mscores1].
    // If your engine order differs, fetch names via engine->getIOTensorName(i) and arrange accordingly.
    std::vector<void*> bindings;
    bindings.resize(8);
    bindings[0] = d_kpts0;
    bindings[1] = d_kpts1;
    bindings[2] = d_desc0;
    bindings[3] = d_desc1;
    bindings[4] = d_matches0;
    bindings[5] = d_matches1;
    bindings[6] = d_mscores0;
    bindings[7] = d_mscores1;

    // Execute
    if (!context->executeV2(bindings.data())) {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }

    // D2H
    std::vector<int64_t> matches0(N0);
    std::vector<int64_t> matches1(N1);
    std::vector<float> mscores0(N0);
    std::vector<float> mscores1(N1);

    if (!cudaCheck(cudaMemcpy(matches0.data(), d_matches0, sz_matches0, cudaMemcpyDeviceToHost), "D2H matches0") ||
        !cudaCheck(cudaMemcpy(matches1.data(), d_matches1, sz_matches1, cudaMemcpyDeviceToHost), "D2H matches1") ||
        !cudaCheck(cudaMemcpy(mscores0.data(), d_mscores0, sz_mscores0, cudaMemcpyDeviceToHost), "D2H mscores0") ||
        !cudaCheck(cudaMemcpy(mscores1.data(), d_mscores1, sz_mscores1, cudaMemcpyDeviceToHost), "D2H mscores1")) {
        return -1;
    }

    // Print a few entries
    std::cout << "N0=" << N0 << ", N1=" << N1 << std::endl;
    std::cout << "matches0[0..4]: ";
    for (int i = 0; i < std::min(5, N0); ++i) std::cout << matches0[i] << " ";
    std::cout << "\nmscores0[0..4]: ";
    for (int i = 0; i < std::min(5, N0); ++i) std::cout << mscores0[i] << " ";
    std::cout << "\nmatches1[0..4]: ";
    for (int i = 0; i < std::min(5, N1); ++i) std::cout << matches1[i] << " ";
    std::cout << "\nmscores1[0..4]: ";
    for (int i = 0; i < std::min(5, N1); ++i) std::cout << mscores1[i] << " ";
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_kpts0);
    cudaFree(d_kpts1);
    cudaFree(d_desc0);
    cudaFree(d_desc1);
    cudaFree(d_matches0);
    cudaFree(d_matches1);
    cudaFree(d_mscores0);
    cudaFree(d_mscores1);

    return 0;
}
