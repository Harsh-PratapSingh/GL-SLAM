#pragma once
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "superpoint.h"

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class LightGlueTRT {
public:
    struct Result {
        std::vector<int64_t> matches0;
        std::vector<int64_t> matches1;
        std::vector<float> mscores0;
        std::vector<float> mscores1;
    };

    LightGlueTRT();
    ~LightGlueTRT();

    // Initialize with ONNX and engine paths. If engine exists, loads it; otherwise builds and saves.
    // Returns true on success.
    bool init(const std::string& onnxPath, const std::string& enginePath);

    // Run inference given host input buffers.
    // kpts*: [N*, 2] row-major floats; desc*: [N*, 256] row-major floats.
    // Returns true on success and fills 'out' with outputs sized to N0/N1.
    bool runInference(const std::vector<float>& kpts0,
                      const std::vector<float>& desc0,
                      const std::vector<float>& kpts1,
                      const std::vector<float>& desc1,
                      int N0, int N1,
                      Result& out);

    LightGlueTRT::Result run_Direct_Inference(SuperPointTRT::Result& spRes0, SuperPointTRT::Result& spRes1);

    // Optional: change workspace cap (default 1GB) before init
    void setWorkspaceSizeBytes(size_t bytes);

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TRT] " << msg << std::endl;
            }
        }
    };

    bool buildAndSaveEngine(const std::string& onnxPath, const std::string& enginePath);
    bool loadEngineFromFile(const std::string& enginePath);

    // Helpers
    bool cudaOK(cudaError_t e);
    bool setInputShapes(int N0, int N1);

    static void toFloatKpts(const std::vector<int64_t>& kptsIntXY, int N, std::vector<float>& kptsFloatXY, int imgWidth, int imgHeight);
    static void sliceDescriptors(const std::vector<float>& descAll, int N, std::vector<float>& descOut);


private:
    Logger logger_;
    size_t workspaceBytes_ = (1ULL << 30); // 1GB default

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Binding indices cache (resolved after engine_ is ready)
    int idx_kpts0_ = -1;
    int idx_kpts1_ = -1;
    int idx_desc0_ = -1;
    int idx_desc1_ = -1;
    int idx_matches0_ = -1;
    int idx_matches1_ = -1;
    int idx_mscores0_ = -1;
    int idx_mscores1_ = -1;
};
