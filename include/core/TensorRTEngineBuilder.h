/**
 * @file TRTEngineConverter.h
 * @brief Converts ONNX models to TensorRT engine files
 */
#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <string>
#include <fstream>

class TensorRTEngineBuilder {
public:
    TensorRTEngineBuilder() = default;
    ~TensorRTEngineBuilder() = default;

    /**
     * @brief Converts an ONNX model to a TensorRT engine file
     * @param onnxPath Path to input ONNX file
     * @param enginePath Path to output TensorRT engine file
     * @return true if conversion succeeded, false otherwise
     */
    bool BuildEngine(const std::string& onnxPath, const std::string& enginePath);

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(nvinfer1::ILogger::Severity severity, 
                const char* msg) noexcept override;
    } m_logger;
};
