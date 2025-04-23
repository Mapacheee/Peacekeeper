#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <tokenizers.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <string>

using namespace tokenizers;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "[!] ";
            if (severity == Severity::kINTERNAL_ERROR) std::cerr << "ERROR: ";
            std::cerr << msg << std::endl;
        }
    }
};

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

#define CHECK_CUDA(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "cuda Error: " << cudaGetErrorString(status) \
                  << " at " << #call << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class TokenizerWrapper {
public:
    TokenizerWrapper(const std::string& tokenizer_path) {
        tok = Tokenizer::FromFile(tokenizer_path);
        validate_special_tokens();
    }

    std::vector<int> encode(const std::string& text, size_t max_length = 512) {
        Encoding encoding = tok->Encode(text);

        if(encoding.GetIds().size() > max_length) {
            encoding.Truncate(max_length);
        } else {
            encoding.Pad(max_length, pad_token_id, bos_token_id, eos_token_id);
        }

        return encoding.GetIds();
    }

private:
    void validate_special_tokens() {
        pad_token_id = tok->TokenToId("[PAD]");
        bos_token_id = tok->TokenToId("<s>");
        eos_token_id = tok->TokenToId("</s>");
        unk_token_id = tok->TokenToId("<unk>");

        if(bos_token_id == -1 || eos_token_id == -1) {
            throw std::runtime_error("special tokens missing in tokenizer");
        }
    }

    std::unique_ptr<Tokenizer> tok;
    int pad_token_id;
    int bos_token_id;
    int eos_token_id;
    int unk_token_id;
};

std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string& enginePath, nvinfer1::IRuntime* runtime) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) throw std::runtime_error("model file not found: " + enginePath);

    file.seekg(0, std::ios::end);
    const size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);

    if (!file) throw std::runtime_error("error reading model file");

    return std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), fileSize)
    );
}

void preprocessImage(const cv::Mat& input, cv::Mat& output, const std::vector<int>& dimensions) {
    cv::cvtColor(input, output, cv::COLOR_BGR2RGB);
    cv::resize(output, output, cv::Size(dimensions[3], dimensions[2]));

    output.convertTo(output, CV_32FC3, 1.0f/255.0f);
    cv::subtract(output, cv::Scalar(0.485f, 0.456f, 0.406f), output);
    cv::divide(output, cv::Scalar(0.229f, 0.224f, 0.225f), output);

    cv::dnn::blobFromImage(output, output);
}

std::vector<int> prepare_input_prompt(TokenizerWrapper& tokenizer, const std::string& user_query) {
    const std::string prompt_template =
        "<s> [INST] <<SYS>>\n"
        "Analiza la imagen y determina si hay violencia. Responde con formato: "
        "[[true/false], descripción en español]. Sé preciso.\n"
        "<</SYS>>\n\n"
        "<image>\n{} [/INST]";

    try {
        return tokenizer.encode(fmt::format(prompt_template, user_query), 512);
    } catch (const std::exception& e) {
        std::cerr << "yokenization error: " << e.what() << std::endl;
        return {};
    }
}

int main() {
    Logger logger;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    try {
        TokenizerWrapper tokenizer("llava_tokenizer.json");

        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        auto engine = loadEngine("ivy_vl_llava.trt", runtime.get());
        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

        const int imageIndex = engine->getBindingIndex("image");
        const int textIndex = engine->getBindingIndex("input_ids");
        const int outputIndex = engine->getBindingIndex("output");

        const auto imageDims = engine->getBindingDimensions(imageIndex);
        const auto textDims = engine->getBindingDimensions(textIndex);
        const auto outputDims = engine->getBindingDimensions(outputIndex);

        const size_t imageSize = imageDims.d[1] * imageDims.d[2] * imageDims.d[3] * sizeof(float);
        const size_t textSize = textDims.d[1] * sizeof(int);
        const size_t outputSize = outputDims.d[1] * sizeof(float);

        std::shared_ptr<float> deviceImage(static_cast<float*>(nullptr), CudaDeleter());
        std::shared_ptr<int> deviceText(static_cast<int*>(nullptr), CudaDeleter());
        std::shared_ptr<float> deviceOutput(static_cast<float*>(nullptr), CudaDeleter());

        CHECK_CUDA(cudaMalloc(&deviceImage, imageSize));
        CHECK_CUDA(cudaMalloc(&deviceText, textSize));
        CHECK_CUDA(cudaMalloc(&deviceOutput, outputSize));

        cv::Mat image = cv::imread("input.jpg");
        if (image.empty()) throw std::runtime_error("failed to load image");

        cv::Mat processedImage;
        preprocessImage(image, processedImage,
                       {imageDims.d[0], imageDims.d[1], imageDims.d[2], imageDims.d[3]});

        CHECK_CUDA(cudaMemcpyAsync(deviceImage.get(), processedImage.data, imageSize,
                                 cudaMemcpyHostToDevice, stream));

        const std::string user_query = "¿Hay alguna escena violenta en esta imagen?";
        auto inputTokens = prepare_input_prompt(tokenizer, user_query);

        if(inputTokens.empty() || inputTokens.size() != textDims.d[1]) {
            throw std::runtime_error("Invalid tokenized input size");
        }

        CHECK_CUDA(cudaMemcpyAsync(deviceText.get(), inputTokens.data(), textSize,
                                 cudaMemcpyHostToDevice, stream));

        void* bindings[] = {deviceImage.get(), deviceText.get(), deviceOutput.get()};
        if (!context->enqueueV2(bindings, stream, nullptr)) {
            throw std::runtime_error("Inference execution failed");
        }

        std::vector<float> output(outputDims.d[1]);
        CHECK_CUDA(cudaMemcpyAsync(output.data(), deviceOutput.get(), outputSize,
                                 cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        std::cout << "output: ";
        for (const auto& val : output) {
            std::cout << static_cast<int>(val) << " ";
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}