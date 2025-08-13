#include <NvInfer.h>
#include <NvInferRuntime.h>  // For IRuntime and deserialization
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>

// Custom logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};

// Function to load ONNX model and build TensorRT engine
std::shared_ptr<nvinfer1::ICudaEngine> buildEngine(const std::string& onnxFile, Logger& logger) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) return nullptr;

    // Use 0 for network flags since kEXPLICIT_BATCH is deprecated
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) return nullptr;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return nullptr;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) return nullptr;

    auto parsed = parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) return nullptr;

    // Find the keypoints output tensor by name
    nvinfer1::ITensor* keypointsTensor = nullptr;
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto output = network->getOutput(i);
        if (std::string(output->getName()) == "keypoints") {
            keypointsTensor = output;
            break;
        }
    }
    if (!keypointsTensor) {
        std::cerr << "Failed to find keypoints output" << std::endl;
        return nullptr;
    }

    // Add a shape layer to get the shape of keypoints [1, num, 2]
    auto shapeLayer = network->addShape(*keypointsTensor);
    auto shapeTensor = shapeLayer->getOutput(0);

    // Slice to get the second dimension (num)
    nvinfer1::Dims start{1, {1}};
    nvinfer1::Dims size{1, {1}};
    nvinfer1::Dims stride{1, {1}};
    auto sliceLayer = network->addSlice(*shapeTensor, start, size, stride);
    auto numTensor = sliceLayer->getOutput(0);

    // Mark as output
    numTensor->setName("num_keypoints");
    numTensor->setType(nvinfer1::DataType::kINT32);
    network->markOutput(*numTensor);

    // Set optimization profile for dynamic shapes
    auto profile = builder->createOptimizationProfile();
    // Use input name "image" based on model description
    nvinfer1::Dims minDims{4, {1, 1, 120, 160}};  // Minimum dimensions (adjust based on model requirements)
    nvinfer1::Dims optDims{4, {1, 1, 480, 640}};  // Optimal dimensions
    nvinfer1::Dims maxDims{4, {1, 1, 1080, 1920}};  // Maximum dimensions
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions("image", nvinfer1::OptProfileSelector::kMAX, maxDims);
    config->addOptimizationProfile(profile);

    // Use setMemoryPoolLimit instead of setMaxWorkspaceSize
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1GB workspace
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    return engine;
}

// Function to load serialized engine from file
std::shared_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string& engineFile, Logger& logger) {
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file) return nullptr;

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return nullptr;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) return nullptr;

    return std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), size));
}

// Function to save engine to file
void saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& engineFile) {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine.serialize());
    if (!serialized) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return;
    }

    std::ofstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file for writing" << std::endl;
        return;
    }

    file.write(static_cast<const char*>(serialized->data()), serialized->size());
}

int main() {
    // Hardcoded paths since no args are needed for the executable
    std::string onnxFile = "superpoint_2048.onnx";  // Replace with your actual ONNX file path
    std::string imagePath = "temp1.png";  // Replace with your actual image file path
    std::string engineFile = "superpoint_2048.engine";  // Engine file to save/load

    Logger logger;

    // Load or build TensorRT engine
    auto engine = loadEngine(engineFile, logger);
    if (!engine) {
        std::cout << "Building engine from ONNX..." << std::endl;
        engine = buildEngine(onnxFile, logger);
        if (!engine) {
            std::cerr << "Failed to build engine" << std::endl;
            return -1;
        }
        saveEngine(*engine, engineFile);
        std::cout << "Engine saved to " << engineFile << std::endl;
    } else {
        std::cout << "Engine loaded from " << engineFile << std::endl;
    }

    // Create execution context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return -1;
    }

    // Load and preprocess image with OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    // Resize to a fixed size (adjust based on model; example 480x640)
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(1024, 1024));

    // Normalize to [0,1] float32
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Prepare input tensor: [1,1,H,W]
    std::vector<float> inputData(1 * 1 * resized.rows * resized.cols);
    std::memcpy(inputData.data(), resized.data, inputData.size() * sizeof(float));

    // Set dynamic input shape using setInputShape
    nvinfer1::Dims inputDims{4, {1, 1, resized.rows, resized.cols}};
    if (!context->setInputShape("image", inputDims)) {
        std::cerr << "Failed to set input shape" << std::endl;
        return -1;
    }

    // Allocate device buffers with maximum sizes for dynamic outputs
    const int maxKeypoints = 2048;

    void* d_image = nullptr;
    void* d_keypoints = nullptr;
    void* d_scores = nullptr;
    void* d_descriptors = nullptr;
    void* d_num_keypoints = nullptr;

    // Allocate input buffer using exact shape
    auto imageShape = context->getTensorShape("image");
    size_t imageSize = 1;
    for (int i = 0; i < imageShape.nbDims; ++i) {
        imageSize *= imageShape.d[i];
    }
    if (cudaMalloc(&d_image, imageSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return -1;
    }

    // Allocate output buffers with max size
    if (cudaMalloc(&d_keypoints, 1 * maxKeypoints * 2 * sizeof(int64_t)) != cudaSuccess ||
        cudaMalloc(&d_scores, 1 * maxKeypoints * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_descriptors, 1 * maxKeypoints * 256 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_num_keypoints, sizeof(int32_t)) != cudaSuccess) {
        std::cerr << "Failed to allocate output buffers" << std::endl;
        return -1;
    }

    // Copy input to device
    cudaMemcpy(d_image, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Prepare bindings array for executeV2 (order: image, keypoints, scores, descriptors, num_keypoints)
    std::vector<void*> bindings = {d_image, d_keypoints, d_scores, d_descriptors};

    // Run inference
    bool status = context->executeV2(bindings.data());
    if (!status) {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }

    // Get actual number of keypoints from the new output
    int32_t numKeypoints = 0;
    cudaMemcpy(&numKeypoints, d_num_keypoints, sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::cout << "keypoints " << d_keypoints;

    if (numKeypoints < 0 || numKeypoints > maxKeypoints) {
        std::cerr << "Invalid number of keypoints: " << numKeypoints << std::endl;
        return -1;
    }

    // Allocate host buffers for outputs with actual size
    std::vector<int64_t> keypoints(4096);
    std::vector<float> scores(2048);
    std::vector<float> descriptors(2048 * 256);

    // Copy outputs from device (using actual size)
    cudaMemcpy(keypoints.data(), d_keypoints, keypoints.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(descriptors.data(), d_descriptors, descriptors.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some results (example: first few keypoints)
    std::cout << "Detected " << numKeypoints << " keypoints (up to 2048)" << std::endl;
    for (int i = 0; i < 1500; ++i) {
        std::cout << "Keypoint " << i << ": (" << keypoints[i * 2] << ", " << keypoints[i * 2 + 1]
                  << ") Score: " << scores[i] << std::endl;
    }

    // Clean up
    cudaFree(d_image);
    cudaFree(d_keypoints);
    cudaFree(d_scores);
    cudaFree(d_descriptors);
    cudaFree(d_num_keypoints);

    return 0;
}