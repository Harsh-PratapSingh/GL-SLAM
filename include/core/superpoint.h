#pragma once
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

class SuperPointTRT {
public:
    struct Result {
        // Up to maxKeypoints entries returned; numValid indicates how many are valid (based on threshold)
        std::vector<int64_t> keypoints;   // length = maxKeypoints*2 (x,y) interleaved
        std::vector<float>   scores;      // length = maxKeypoints
        std::vector<float>   descriptors; // length = maxKeypoints*256
        int numValid = 0;                 // number of valid keypoints (scores > threshold)
    };

    SuperPointTRT();
    ~SuperPointTRT();

    // Optional: change workspace cap (default 1GB) before init
    void setWorkspaceSizeBytes(size_t bytes);

    // Optional: change max keypoints (default 2048) before init
    void setMaxKeypoints(int maxKpts);

    // Optional: change threshold used to count valid keypoints (default 0.0f) before runInference
    void setScoreThreshold(float thr);

    // Initialize with ONNX and engine paths. If engine exists, loads it; otherwise builds and saves.
    // min/opt/max are the static input shape used to build the engine (H,W must be equal for all three).
    // Returns true on success.
    bool init(const std::string& onnxPath,
              const std::string& enginePath,
              int height = 376, int width = 1241);

    // Run inference on a single-channel float32 image [H,W] normalized to [0,1].
    // The method internally sets input shape [1,1,H,W], allocates device buffers,
    // runs inference, copies outputs, and counts valid keypoints by scoreThreshold_.
    // Returns true on success and fills 'out'.
    bool runInference(const float* imageData, int height, int width, Result& out);

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                // Print warnings and above
                fprintf(stderr, "[TRT] %s\n", msg);
            }
        }
    };

    // Helpers
    bool buildAndSaveEngine(const std::string& onnxPath,
                            const std::string& enginePath,
                            int height, int width);

    bool loadEngineFromFile(const std::string& enginePath);

    bool cudaOK(cudaError_t e);

private:
    Logger logger_;
    size_t workspaceBytes_ = (1ULL << 30); // 1GB default
    int maxKeypoints_ = 2048;
    float scoreThreshold_ = 0.0f;

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Binding indices (resolved after engine is ready)
    int idx_image_ = -1;
    int idx_keypoints_ = -1;
    int idx_scores_ = -1;
    int idx_descriptors_ = -1;
};
