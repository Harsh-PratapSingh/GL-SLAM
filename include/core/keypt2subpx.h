#pragma once
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "lightglue.h"  // For LightGlueTRT::Result

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class Keypt2SubpxTRT {
public:
    struct Result {
        std::vector<float> refined_keypt0;  // length = N*2 (x,y) interleaved floats
        std::vector<float> refined_keypt1;  // length = N*2 (x,y) interleaved floats
    };

    Keypt2SubpxTRT();
    ~Keypt2SubpxTRT();

    // Initialize with ONNX and engine paths. If engine exists, loads it; otherwise builds and saves.
    // Returns true on success.
    bool init(const std::string& onnxPath, const std::string& enginePath);

    // Run inference given host input buffers.
    // keypt*: [N, 2] row-major floats; desc*: [N, 256] row-major floats; img*: [3*H*W] floats; score*: [N] floats (per-keypoint scores).
    // Returns true on success and fills 'out' with outputs sized to N.
    bool runInference(const std::vector<float>& keypt0,
                      const std::vector<float>& keypt1,
                      const std::vector<float>& img0,
                      const std::vector<float>& img1,
                      const std::vector<float>& desc0,
                      const std::vector<float>& desc1,
                      const std::vector<float>& score0,
                      const std::vector<float>& score1,
                      int N, int H, int W,
                      Result& out);

        Keypt2SubpxTRT::Result run_Direct_Inference(const LightGlueTRT::Result& lgRes,
                                                const cv::Mat& img0,
                                                const cv::Mat& img1);

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
    bool setInputShapes(int N, int H, int W);

    static void extractMatched(const LightGlueTRT::Result& lgRes, std::vector<float>& m_keypt0, std::vector<float>& m_keypt1,
                                    std::vector<float>& m_desc0, std::vector<float>& m_desc1,
                                    std::vector<float>& m_score0, std::vector<float>& m_score1,
                                    int& N, int H, int W);

private:
    Logger logger_;
    size_t workspaceBytes_ = (1ULL << 30); // 1GB default

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Binding indices cache (resolved after engine_ is ready)
    int idx_keypt0_ = -1;
    int idx_keypt1_ = -1;
    int idx_img0_ = -1;
    int idx_img1_ = -1;
    int idx_desc0_ = -1;
    int idx_desc1_ = -1;
    int idx_score0_ = -1;
    int idx_score1_ = -1;
    int idx_refined_keypt0_ = -1;
    int idx_refined_keypt1_ = -1;
};
