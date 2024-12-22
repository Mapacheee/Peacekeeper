#include "Logger.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

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
        const int inputImageIndex = engine->getBindingIndex("image");
        const int inputTextIndex = engine->getBindingIndex("input_ids");
        const int outputIndex = engine->getBindingIndex("output");

        nvinfer1::Dims imageDims = engine->getBindingDimensions(inputImageIndex);
        nvinfer1::Dims textDims = engine->getBindingDimensions(inputTextIndex);
        nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

        size_t imageSize = batchSize * imageDims.d[1] * imageDims.d[2] * imageDims.d[3] * sizeof(float);
        size_t textSize = batchSize * textDims.d[1] * sizeof(int);
        size_t outputSize = batchSize * outputDims.d[1] * sizeof(float);

        void* deviceImage = nullptr;
        void* deviceText = nullptr;
        void* deviceOutput = nullptr;
        cudaMalloc(&deviceImage, imageSize);
        cudaMalloc(&deviceText, textSize);
        cudaMalloc(&deviceOutput, outputSize);

        cv::Mat image = cv::imread("input.jpg");
        std::vector<float> inputImageBuffer(imageSize / sizeof(float));
        preprocessImage(image, inputImageBuffer.data(), imageDims.d[2], imageDims.d[3]);

        cudaMemcpy(deviceImage, inputImageBuffer.data(), imageSize, cudaMemcpyHostToDevice);

        std::vector<int> inputTextBuffer(textSize / sizeof(int), 101); // Token [CLS]
        cudaMemcpy(deviceText, inputTextBuffer.data(), textSize, cudaMemcpyHostToDevice);

        void* bindings[] = {deviceImage, deviceText, deviceOutput};
        context->enqueue(batchSize, bindings, 0, nullptr);

        std::vector<float> outputBuffer(outputSize / sizeof(float));
        cudaMemcpy(outputBuffer.data(), deviceOutput, outputSize, cudaMemcpyDeviceToHost);

        for (float val : outputBuffer) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        cudaFree(deviceImage);
        cudaFree(deviceText);
        cudaFree(deviceOutput);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}