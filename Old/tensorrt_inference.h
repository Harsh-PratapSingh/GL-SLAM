#pragma once

#include <vector>
#include <string>
#include <memory>

class TensorRTInference {
public:
    TensorRTInference(const std::string& onnxFile, const std::string& engineFile);
    ~TensorRTInference();

    // Run inference and return keypoints, matches, and scores
    bool runInference(const std::string& leftImage, const std::string& rightImage,
                      std::vector<int64_t>& keypoints, std::vector<int64_t>& matches,
                      std::vector<float>& scores);

private:
    void* engineData_; // Store engine data as raw pointer
    size_t engineSize_;
    void* inputDevice_;
    void* keypointsDevice_;
    void* matchesDevice_;
    void* scoresDevice_;

    // TensorRT objects (opaque pointers managed internally)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
