#include "core/TensorRTEngineBuilder.h"
#include <iostream>

void TensorRTEngineBuilder::Logger::log(
    nvinfer1::ILogger::Severity severity, 
    const char* msg) noexcept 
{
    if (severity <= nvinfer1::ILogger::Severity::kINFO) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

bool TensorRTEngineBuilder::BuildEngine(
    const std::string& onnxPath, 
    const std::string& enginePath) 
{
    using namespace nvinfer1;

    // 1. Create builder
    IBuilder* builder = createInferBuilder(m_logger);
    if (!builder) return false;

    // 2. Create network with explicit batch
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(
        NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // 3. Parse ONNX model
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, m_logger);
    if (!parser->parseFromFile(onnxPath.c_str(), 2)) {
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // 4. Configure builder
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kFP16);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);  // 1GB

    // 5. Set input profile
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    ITensor* input = network->getInput(0);
    const Dims inputDims = input->getDimensions();
    
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, inputDims);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, inputDims);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, inputDims);
    config->addOptimizationProfile(profile);

    // 6. Build and save engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    IHostMemory* serializedEngine = engine->serialize();
    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), 
                   serializedEngine->size());

    // Cleanup
    delete serializedEngine;
    delete engine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

