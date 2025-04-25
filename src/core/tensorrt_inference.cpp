#include "core/tensorrt_inference.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

// Logger class for TensorRT
class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cerr << msg << std::endl;
    }
  }
};

// Utility to check CUDA errors
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw std::runtime_error("CUDA error occurred");                         \
    }                                                                          \
  } while (0)

// Pimpl idiom to hide TensorRT implementation details
struct TensorRTInference::Impl {
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
};

// Load or build engine
static std::vector<char> loadOrBuildEngine(const std::string& onnxFile,
                                           const std::string& engineFile,
                                           Logger& logger) {
  std::ifstream engineStream(engineFile, std::ios::binary);
  if (engineStream.good()) {
    engineStream.seekg(0, std::ios::end);
    size_t size = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    engineStream.read(engineData.data(), size);
    engineStream.close();
    std::cout << "Loaded existing engine from " << engineFile << std::endl;
    return engineData;
  }

  std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
  std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0U));
  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, logger));

  std::ifstream onnxStream(onnxFile, std::ios::binary);
  onnxStream.seekg(0, std::ios::end);
  size_t onnxSize = onnxStream.tellg();
  onnxStream.seekg(0, std::ios::beg);
  std::vector<char> onnxData(onnxSize);
  onnxStream.read(onnxData.data(), onnxSize);
  onnxStream.close();

  if (!parser->parse(onnxData.data(), onnxSize)) {
    std::cerr << "ERROR: Failed to parse ONNX file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

  std::unique_ptr<ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    std::cerr << "ERROR: Failed to build engine" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::unique_ptr<IHostMemory> serializedEngine(engine->serialize());
  std::ofstream engineOut(engineFile, std::ios::binary);
  engineOut.write(static_cast<char*>(serializedEngine->data()),
                  serializedEngine->size());
  engineOut.close();
  std::cout << "Built and saved engine to " << engineFile << std::endl;

  return std::vector<char>(static_cast<char*>(serializedEngine->data()),
                           static_cast<char*>(serializedEngine->data()) +
                               serializedEngine->size());
}

// Load and preprocess image
static bool loadAndPreprocessImage(const std::string& filePath, float* buffer,
                                   int offset, int height, int width) {
  cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cerr << "ERROR: Failed to load image " << filePath << std::endl;
    return false;
  }

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      buffer[offset + h * width + w] =
          static_cast<float>(resized.at<uint8_t>(h, w)) / 255.0f;
    }
  }
  return true;
}

TensorRTInference::TensorRTInference(const std::string& onnxFile,
                                     const std::string& engineFile) {
  Logger logger;
  auto engineData = loadOrBuildEngine(onnxFile, engineFile, logger);
  engineSize_ = engineData.size();
  engineData_ = malloc(engineSize_);
  memcpy(engineData_, engineData.data(), engineSize_);

  impl_ = std::make_unique<Impl>();
  impl_->runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
  impl_->engine = std::unique_ptr<ICudaEngine>(
      impl_->runtime->deserializeCudaEngine(engineData_, engineSize_));
  if (!impl_->engine) {
    std::cerr << "ERROR: Failed to deserialize engine" << std::endl;
    throw std::runtime_error("Engine deserialization failed");
  }

  impl_->context = std::unique_ptr<IExecutionContext>(
      impl_->engine->createExecutionContext());
  if (!impl_->context) {
    std::cerr << "ERROR: Failed to create execution context" << std::endl;
    throw std::runtime_error("Context creation failed");
  }

  // Allocate device memory
  const int batchSize = 2;
  const int channels = 1;
  const int height = 1024;
  const int width = 1024;
  const int inputSize = batchSize * channels * height * width;
  const int keypointsSize = batchSize * 1024 * 2;
  const int maxMatches = 4096;

  CHECK_CUDA(cudaMalloc(&inputDevice_, inputSize * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&keypointsDevice_, keypointsSize * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&matchesDevice_, maxMatches * 3 * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&scoresDevice_, maxMatches * sizeof(float)));
}

TensorRTInference::~TensorRTInference() {
  cudaFree(inputDevice_);
  cudaFree(keypointsDevice_);
  cudaFree(matchesDevice_);
  cudaFree(scoresDevice_);
  free(engineData_);
}

bool TensorRTInference::runInference(const std::string& leftImage,
                                     const std::string& rightImage,
                                     std::vector<int64_t>& keypoints,
                                     std::vector<int64_t>& matches,
                                     std::vector<float>& scores) {
  const int batchSize = 2;
  const int channels = 1;
  const int height = 1024;
  const int width = 1024;
  const int inputSize = batchSize * channels * height * width;
  const int keypointsSize = batchSize * 1024 * 2;
  const int maxMatches = 4096;

  // Prepare input
  std::vector<float> inputHost(inputSize);
  if (!loadAndPreprocessImage(leftImage, inputHost.data(), 0, height, width) ||
      !loadAndPreprocessImage(rightImage, inputHost.data(), height * width, height, width)) {
    return false;
  }

  CHECK_CUDA(
      cudaMemcpy(inputDevice_, inputHost.data(), inputSize * sizeof(float),
                 cudaMemcpyHostToDevice));

  // Bindings
  void* bindings[4] = {inputDevice_, keypointsDevice_, matchesDevice_, scoresDevice_};

  // Run inference
  if (!impl_->context->executeV2(bindings)) {
    std::cerr << "ERROR: Failed to execute inference" << std::endl;
    return false;
  }

  // Resize output vectors
  keypoints.resize(keypointsSize);
  matches.resize(maxMatches * 3);
  scores.resize(maxMatches);

  // Copy outputs to host
  CHECK_CUDA(cudaMemcpy(keypoints.data(), keypointsDevice_,
                        keypointsSize * sizeof(int64_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(matches.data(), matchesDevice_,
                        maxMatches * 3 * sizeof(int64_t), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(scores.data(), scoresDevice_,
                        maxMatches * sizeof(float), cudaMemcpyDeviceToHost));

  return true;
}