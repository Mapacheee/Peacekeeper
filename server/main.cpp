#include "Logger.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <crow.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string& enginePath, nvinfer1::IRuntime* runtime) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("tensor rt not found: " + enginePath);
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);

    return std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fileSize));
}

void preprocessImage(const cv::Mat& image, float* inputBuffer, int inputHeight, int inputWidth) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(resized, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    cv::merge(channels, 3, resized);

    std::memcpy(inputBuffer, resized.data, inputHeight * inputWidth * 3 * sizeof(float));
}

int main() {

    Logger logger;

    try {
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        std::unique_ptr<nvinfer1::ICudaEngine> engine = loadEngine("ivy_vl_llava.trt", runtime.get());
        std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};


        const int batchSize = 1;
        const int inputIndex = engine->getBindingIndex("input");
        const int outputIndex = engine->getBindingIndex("output");
        const nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
        const nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

        size_t inputSize = batchSize * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
        size_t outputSize = batchSize * outputDims.d[1] * sizeof(float);

        void* deviceInput = nullptr;
        void* deviceOutput = nullptr;
        cudaMalloc(&deviceInput, inputSize);
        cudaMalloc(&deviceOutput, outputSize);


        crow::SimpleApp app;

        CROW_ROUTE(app, "/process").methods("POST"_method)([&](const crow::request& req) {
            try {
                std::vector<uchar> data(req.body.begin(), req.body.end());
                cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);
                if (image.empty()) {
                    return crow::response(400, "image not valid");
                }

                std::vector<float> inputBuffer(inputSize / sizeof(float));
                preprocessImage(image, inputBuffer.data(), inputDims.d[2], inputDims.d[3]);

                cudaMemcpy(deviceInput, inputBuffer.data(), inputSize, cudaMemcpyHostToDevice);

                void* bindings[] = {deviceInput, deviceOutput};
                context->enqueue(batchSize, bindings, 0, nullptr);

                std::vector<float> outputBuffer(outputSize / sizeof(float));
                cudaMemcpy(outputBuffer.data(), deviceOutput, outputSize, cudaMemcpyDeviceToHost);

                std::ostringstream response;
                for (const auto& val : outputBuffer) {
                    response << val << " ";
                }
                return crow::response(200, response.str());
            } catch (const std::exception& e) {
                return crow::response(500, std::string("error: ") + e.what());
            }
        });

        std::cout << "server started at http://localhost:8080" << std::endl;
        app.port(8080).multithreaded().run();


        cudaFree(deviceInput);
        cudaFree(deviceOutput);
    }
    catch (const std::exception& e) {
        std::cerr << "error when loading tensor rt: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}