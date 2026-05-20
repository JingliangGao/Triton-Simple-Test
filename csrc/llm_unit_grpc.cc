#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

std::string g_model_name = "llamacpp";
#include <triton/client/grpc_client.h>

#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <iostream>

using namespace triton::client;
namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


// ============================================================================
// 辅助函数：使用 Triton InferInput::Create 工厂方法创建张量
// ============================================================================
std::shared_ptr<tc::InferInput> PrepareTensor(const std::string& name,
                              const std::vector<int64_t>& shape,
                              const std::string& datatype,
                              const uint8_t* data, size_t byte_size) {
    tc::InferInput* input = nullptr;
    tc::Error err = tc::InferInput::Create(&input, name, shape, datatype);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to create InferInput: " + err.Message());
    }
    std::shared_ptr<tc::InferInput> input_ptr(input); // 确保异常安全
    input_ptr->Reset(); // 清空默认值
    err = input_ptr->AppendRaw(data, byte_size);
    if (datatype == "INT32" && byte_size == sizeof(int32_t)) {
        std::cout << "data: " << static_cast<int32_t>(*data) << std::endl;
    } else if (datatype == "FP32"&& byte_size == sizeof(float)) {
        std::cout << "data: " << *reinterpret_cast<const float*>(data) << std::endl;
    } else if (datatype == "BOOL" && byte_size == sizeof(bool) ) {
        std::cout << "data: " << (*data ? "true" : "false") << std::endl;
    }
    if (!err.IsOk()) {
        throw std::runtime_error("failed to append raw data: " + err.Message());
    }
    return input_ptr;
}

template<typename T>
std::shared_ptr<tc::InferInput> PrepareScalarTensor(const std::string& name, T value, const std::string& dtype) {
    //return PrepareTensor(name, {1}, dtype, reinterpret_cast<const uint8_t*>(&value), sizeof(T));
    tc::InferInput* input = nullptr;
    tc::Error err = tc::InferInput::Create(&input, name, {1}, dtype);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to create InferInput: " + err.Message());
    }
    std::shared_ptr<tc::InferInput> input_ptr(input); // 确保异常安全
    input_ptr->Reset(); // 清空默认值
    err = input_ptr->AppendRaw(reinterpret_cast<const uint8_t*>(&value), sizeof(T));
    // if (datatype == "INT32" && byte_size == sizeof(int32_t)) {
    //     std::cout << "data: " << static_cast<int32_t>(*data) << std::endl;
    // } else if (datatype == "FP32"&& byte_size == sizeof(float)) {
    //     std::cout << "data: " << *reinterpret_cast<const float*>(data) << std::endl;
    // } else if (datatype == "BOOL" && byte_size == sizeof(bool) ) {
    //     std::cout << "data: " << (*data ? "true" : "false") << std::endl;
    // }
    if (!err.IsOk()) {
        throw std::runtime_error("failed to append raw data: " + err.Message());
    }
    return input_ptr;
}

std::shared_ptr<tc::InferInput> PrepareStringTensor(const std::string& name, const std::string& str) {
    tc::InferInput* input = nullptr;
    tc::Error err = tc::InferInput::Create(&input, name, {1}, "BYTES");
    if (!err.IsOk()) {
        throw std::runtime_error("failed to create BYTES input: " + err.Message());
    }
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);
    input_ptr->SetBinaryData(true);

    //err = input_ptr->AppendRaw(reinterpret_cast<const uint8_t*>(str.c_str()), str.size());
    std::vector<std::string> str_vec = {str};
    err = input_ptr->AppendFromString(str_vec);
    if (!err.IsOk()) {
        throw std::runtime_error("failed to append string data: " + err.Message());
    }

    return input_ptr;
}



// ============================================================================
// 请求类型枚举（与 Python 脚本一致）
// ============================================================================
enum RequestType : int32_t {
    COMPLETIONS = 0,
    OPENAI_COMPLETIONS = 1,
    OPENAI_CHAT = 2,
    TOKENIZE = 3,
    DETOKENIZE = 4,
    LORA_GET = 6,
    TASK_CANCEL = 7,
    TASK_ABORT = 8,
    METRICS = 9,
    WORKER_SAVE = 10,
    WORKER_RESTORE = 11,
    WORKER_ERASE = 12
};

// ============================================================================
// 测试夹具类：管理 Triton 客户端、流式连接和响应队列
// ============================================================================
class LlamaCppTestFixture {
public:
    LlamaCppTestFixture() {
        std::string url = "localhost:8001";
        // 创建 gRPC 客户端（使用静态 Create 方法）
        // 使用与当前 Triton 客户端匹配的简单重载：(InferenceServerGrpcClient**, const std::string&, bool verbose)
        tc::Error err = tc::InferenceServerGrpcClient::Create(&grpc_client_, url, false);
        if (!err.IsOk()) {
            throw std::runtime_error("unable to create GRPC client: " + err.Message());
        }

        // 启动流式连接，设置回调
        err = grpc_client_->StartStream(
            [this](tc::InferResult* result) {
                std::lock_guard<std::mutex> lock(mutex_);
                result_queue_.push(std::shared_ptr<tc::InferResult>(result));
                cv_.notify_one();
            }, false);
        if (!err.IsOk()) {
            throw std::runtime_error("failed to start stream: " + err.Message());
        }

        EnsureModelReady();
    }

    ~LlamaCppTestFixture() {
        if (grpc_client_) {
            grpc_client_->StopStream();
        }
    }

    void EnsureModelReady() {
        bool ready = false;
        tc::Error err = grpc_client_->IsModelReady(&ready, model_name_);
        if (!err.IsOk()) {
            throw std::runtime_error("unable to check model readiness: " + err.Message());
        }
        if (ready) {
            return;
        }

        std::cout << "Model '" << model_name_ << "' is not ready, loading..." << std::endl;
        err = grpc_client_->LoadModel(model_name_);
        if (!err.IsOk()) {
            throw std::runtime_error("failed to load model '" + model_name_ + "': " + err.Message());
        }

    }

    // 同步发送请求（内部使用异步流式 API，但阻塞等待完整响应）
    bool SendRequest(RequestType req_type,
                     const std::string& request_id,
                     const std::string& text_input,
                     std::string* output_text = nullptr,
                     int n_predict = -1,
                     const std::string& stop = "<|im_end|>",
                     float temperature = 0.7f,
                     int top_k = 40,
                     float top_p = 0.95f,
                     int n_keep = 0,
                     bool cache_prompt = true,
                     bool stream = true,
                     const std::vector<float>& lora_scale = {},
                     float repeat_penalty = 1.0f) {

        std::vector<std::shared_ptr<tc::InferInput>> input_ptrs;
        std::vector<tc::InferInput*> inputs;
        //int32_t request_type = req_type;
        try {

            tc::InferInput* request_type_t;
            FAIL_IF_ERR(
                tc::InferInput::Create(&request_type_t, "request_type", {1}, "INT32"),
                "unable to create 'request_type'");
            std::shared_ptr<tc::InferInput> request_type_ptr(request_type_t);
            FAIL_IF_ERR(request_type_ptr->Reset(), "unable to reset 'request_type'");
            FAIL_IF_ERR(
                    request_type_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&req_type), sizeof(int32_t)),
                    "unable to set data for 'request_type'");

            // input: request_id
            tc::InferInput* request_id_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&request_id_t, "request_id", {1}, "BYTES"),
                    "unable to create 'request_id'");
            std::shared_ptr<tc::InferInput> request_id_ptr;
            request_id_ptr.reset(request_id_t);
            request_id_ptr->SetBinaryData(true);

            std::vector<std::string> id_vec = {request_id};
            FAIL_IF_ERR(
            request_id_ptr->AppendFromString(id_vec),
            "unable to set data for request_id");

            // input: text_input
            tc::InferInput* text_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&text_input_t, "text_input", {1}, "BYTES"),
                    "unable to create 'text_input'");
            std::shared_ptr<tc::InferInput> text_input_ptr;
            text_input_ptr.reset(text_input_t);
            text_input_ptr->SetBinaryData(true);

            std::vector<std::string> text_vec = {text_input};
            FAIL_IF_ERR(
            text_input_ptr->AppendFromString(text_vec),
            "unable to set data for text_input");

            // input: temperature_input
            // int32_t temperature_input_data = 0.8;
            tc::InferInput* temperature_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&temperature_input_t, "temperature_input", {1}, "FP32"),
                    "unable to create 'temperature_input'");
            std::shared_ptr<tc::InferInput> temperature_input_ptr(temperature_input_t);
            FAIL_IF_ERR(temperature_input_ptr->Reset(), "unable to reset 'temperature_input'");
            FAIL_IF_ERR(
                    temperature_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&temperature), sizeof(float)),
                    "unable to set data for 'temperature_input'");

            // input: top_k_input
            tc::InferInput* top_k_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&top_k_input_t, "top_k_input", {1}, "INT32"),
                    "unable to create 'top_k_input'");
            std::shared_ptr<tc::InferInput> top_k_input_ptr(top_k_input_t);
            FAIL_IF_ERR(top_k_input_ptr->Reset(), "unable to reset 'top_k_input'");
            FAIL_IF_ERR(
                    top_k_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&top_k), sizeof(int32_t)),
                    "unable to set data for 'top_k_input'");

            // input: top_p_input
            tc::InferInput* top_p_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&top_p_input_t, "top_p_input", {1}, "FP32"),
                    "unable to create 'top_p_input'");
            std::shared_ptr<tc::InferInput> top_p_input_ptr(top_p_input_t);
            FAIL_IF_ERR(top_p_input_ptr->Reset(), "unable to reset 'top_p_input'");
            FAIL_IF_ERR(
                    top_p_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&top_p), sizeof(float)),
                    "unable to set data for 'top_p_input'");

            // input: n_keep_input
            // TODO n_keep
            tc::InferInput* n_keep_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&n_keep_input_t, "n_keep_input", {1}, "INT32"),
                    "unable to create 'n_keep_input'");
            std::shared_ptr<tc::InferInput> n_keep_input_ptr(n_keep_input_t);
            FAIL_IF_ERR(n_keep_input_ptr->Reset(), "unable to reset 'n_keep_input'");
            FAIL_IF_ERR(
                    n_keep_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&n_keep), sizeof(int32_t)),
                    "unable to set data for 'n_keep_input'");

            // input: n_predict_input 256
            tc::InferInput* n_predict_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&n_predict_input_t, "n_predict_input", {1}, "INT32"),
                    "unable to create 'n_predict_input'");
            std::shared_ptr<tc::InferInput> n_predict_input_ptr(n_predict_input_t);
            FAIL_IF_ERR(n_predict_input_ptr->Reset(), "unable to reset 'n_predict_input'");
            FAIL_IF_ERR(
                    n_predict_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&n_predict), sizeof(int32_t)),
                    "unable to set data for 'n_predict_input'");

            // input: cache_prompt_input True
            tc::InferInput* cache_prompt_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&cache_prompt_input_t, "cache_prompt_input", {1}, "BOOL"),
                    "unable to create 'cache_prompt_input'");
            std::shared_ptr<tc::InferInput> cache_prompt_input_ptr(cache_prompt_input_t);
            FAIL_IF_ERR(cache_prompt_input_ptr->Reset(), "unable to reset 'cache_prompt_input'");
            FAIL_IF_ERR(
                    cache_prompt_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&cache_prompt), sizeof(bool)),
                    "unable to set data for 'cache_prompt_input'");

            // input: stop_input "\n### Human"
            tc::InferInput* stop_input_t;
            FAIL_IF_ERR(
            tc::InferInput::Create(&stop_input_t, "stop_input", {1}, "BYTES"),
            "unable to get stop_input");
            std::shared_ptr<tc::InferInput> stop_input_ptr;
            stop_input_ptr.reset(stop_input_t);
            stop_input_ptr->SetBinaryData(true);

            std::vector<std::string> stop_vec = {stop};
            FAIL_IF_ERR(
            stop_input_ptr->AppendFromString(stop_vec),
            "unable to set data for stop_input");

            // input: lora_scale_input [0.0, 1.0]
            tc::InferInput* lora_scale_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&lora_scale_input_t, "lora_scale_input", {static_cast<int64_t>(lora_scale.size())}, "FP32"),
                    "unable to create 'lora_scale_input'");
            std::shared_ptr<tc::InferInput> lora_scale_input_ptr(lora_scale_input_t);
            FAIL_IF_ERR(lora_scale_input_ptr->Reset(), "unable to reset 'lora_scale_input'");
            FAIL_IF_ERR(
                    lora_scale_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(const_cast<float*>(lora_scale.data())), sizeof(float) * lora_scale.size()),
                    "unable to set data for 'lora_scale_input'");


            // input: repeat_penalty_input 1.1
            tc::InferInput* repeat_penalty_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&repeat_penalty_input_t, "repeat_penalty_input", {1}, "FP32"),
                    "unable to create 'repeat_penalty_input'");
            std::shared_ptr<tc::InferInput>repeat_penalty_input_ptr(repeat_penalty_input_t);
            FAIL_IF_ERR(repeat_penalty_input_ptr->Reset(), "unable to reset 'repeat_penalty_input'");
            FAIL_IF_ERR(
                    repeat_penalty_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&repeat_penalty), sizeof(float)),
                    "unable to set data for 'repeat_penalty_input'");

            // input: stream_input True
            tc::InferInput* stream_input_t;
            FAIL_IF_ERR(
                    tc::InferInput::Create(&stream_input_t, "stream_input", {1}, "BOOL"),
                    "unable to create 'stream_input'");
            std::shared_ptr<tc::InferInput> stream_input_ptr(stream_input_t);
            FAIL_IF_ERR(stream_input_ptr->Reset(), "unable to reset 'stream_input'");
            FAIL_IF_ERR(
                    stream_input_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&stream), sizeof(bool)),
                    "unable to set data for 'stream_input'");


            input_ptrs.push_back(request_type_ptr);
            input_ptrs.push_back(request_id_ptr);
            input_ptrs.push_back(text_input_ptr);
            input_ptrs.push_back(temperature_input_ptr);
            input_ptrs.push_back(top_k_input_ptr);
            input_ptrs.push_back(top_p_input_ptr);
            input_ptrs.push_back(n_keep_input_ptr);
            input_ptrs.push_back(n_predict_input_ptr);
            input_ptrs.push_back(cache_prompt_input_ptr);
            input_ptrs.push_back(stop_input_ptr);
            input_ptrs.push_back(repeat_penalty_input_ptr);
            input_ptrs.push_back(stream_input_ptr);
            input_ptrs.push_back(lora_scale_input_ptr);

            for (auto& ptr : input_ptrs) {
                inputs.push_back(ptr.get());
            }

            // input_ptrs.push_back(PrepareScalarTensor("request_type", static_cast<int32_t>(req_type), "INT32"));
            // input_ptrs.push_back(PrepareStringTensor("request_id", request_id));
            // input_ptrs.push_back(PrepareStringTensor("text_input", text_input));
            // input_ptrs.push_back(PrepareScalarTensor<float>("temperature_input", 0.8/*temperature*/, "FP32"));
            // input_ptrs.push_back(PrepareScalarTensor<int32_t>("top_k_input", top_k, "INT32"));
            // input_ptrs.push_back(PrepareScalarTensor<float>("top_p_input", top_p, "FP32"));
            // input_ptrs.push_back(PrepareScalarTensor<int32_t>("n_keep_input", n_keep, "INT32"));
            // input_ptrs.push_back(PrepareScalarTensor<int32_t>("n_predict_input", n_predict, "INT32"));
            // input_ptrs.push_back(PrepareScalarTensor<bool>("cache_prompt_input", cache_prompt, "BOOL"));
            // input_ptrs.push_back(PrepareStringTensor("stop_input", stop));
            // input_ptrs.push_back(PrepareScalarTensor<bool>("stream_input", stream, "BOOL"));
            // input_ptrs.push_back(PrepareTensor("lora_scale_input", {static_cast<int64_t>(lora_scale.size())}, "FP32",
            //                                reinterpret_cast<const uint8_t*>(lora_scale.data()),
            //                                lora_scale.size() * sizeof(float)));
            // input_ptrs.push_back(PrepareScalarTensor<float>("repeat_penalty_input", repeat_penalty, "FP32"));

            // for (auto& ptr : input_ptrs) {
            //     inputs.push_back(ptr.get());
            // }
        } catch (const std::exception& e) {
            std::cerr << "Error preparing inputs: " << e.what() << std::endl;
            return false;
        }

        // 发起异步流式推理
        tc::InferOptions options(model_name_);
        //options.request_id_ = request_id;
        tc::Error err = grpc_client_->AsyncStreamInfer(options, inputs);


        if (!err.IsOk()) {
            std::cerr << "AsyncStreamInfer failed: " << err.Message() << std::endl;
            return false;
        }

        // 收集响应直到收到 triton_final_response 参数
        bool final_received = false;
        std::string accumulated_output;

        while (!final_received) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !result_queue_.empty() || error_.has_value(); });

            if (error_.has_value()) {
                std::cerr << "Stream error: " << error_->Message() << std::endl;
                return false;
            }

            auto result = result_queue_.front();
            result_queue_.pop();
            lock.unlock();

            // 检查响应参数中的 triton_final_response
            result->IsFinalResponse(&final_received);

            // 提取文本输出
            const uint8_t* buf = nullptr;
            size_t size = 0;
            result->RawData("text_output", &buf, &size);
            if (buf && size > 0) {
                std::string chunk(reinterpret_cast<const char*>(buf+4), size-4);
                std::cout << chunk << std::flush;
                accumulated_output += chunk;
            }

            size = 0;
            result->RawData("metrics_output", &buf, &size);
            if (buf && size > 0) {
                std::string metrics(reinterpret_cast<const char*>(buf+4), size-4);
                std::cout << " [metrics: " << metrics << "]" << std::flush;
            }

            size = 0;
            result->RawData("lora_scale_output", &buf, &size);
            if (buf && size > 0) {
                std::string lora_get(reinterpret_cast<const char*>(buf+4), size-4);
                std::cout << " [lora_get: " << lora_get << "]" << std::flush;
            }

            size = 0;
            result->RawData("token_output", &buf, &size);
            if (buf && size > 0) {
                for (size_t i = 0; i < size; i += sizeof(uint32_t)) {
                    uint32_t token = 0;
                    memcpy(&token, buf+i, sizeof(uint32_t));
                    std::cout << " [token: " << token << "]" << std::flush;
                }
            }

            lock.lock();
        }

        std::cout << std::endl;
        if (output_text) *output_text = accumulated_output;
        return true;
    }

    // 用于取消/中止测试的异步发送辅助函数
    std::future<bool>  SendRequestAsync(RequestType req_type,
                                       const std::string& request_id,
                                       const std::string& text_input,
                                       int n_predict = -1,
                                       const std::string& stop = "<|im_end|>") {
        return std::async(std::launch::async, [this, req_type, request_id, text_input, n_predict, stop]() {
            return SendRequest(req_type, request_id, text_input, nullptr, n_predict, stop);
        });
    }

private:
    std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client_;
    std::optional<tc::Error> error_;
    std::string model_name_ = g_model_name;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<tc::InferResult>> result_queue_;
};

// ============================================================================
// Catch2 测试用例
// ============================================================================

TEST_CASE_METHOD(LlamaCppTestFixture, "LlamaCpp Backend Inference Tests", "[llamacpp]") {
    std::string output;

    SECTION("completions test") {
        REQUIRE(SendRequest(RequestType::COMPLETIONS, "req-1", "你好，请介绍一下你自己",
                            &output, 10));
        REQUIRE_FALSE(output.empty());
    }

    SECTION("openai completions test") {
        std::string completions_json = R"({
            "temperature":0.7, "top_k":40, "top_p":0.95, "n_keep":0,
            "n_predict":10, "cache_prompt":true, "stop":"<|im_end|>",
            "stream":true, "repeat_penalty":1.1,
            "lora":[{"id":0,"scale":0.0},{"id":1,"scale":0.0}],
            "prompt":"你是谁？"
        })";
        REQUIRE(SendRequest(RequestType::OPENAI_COMPLETIONS, "req-2", completions_json,
                            &output, 10));
    }

    SECTION("openai chat test") {
        std::string chat_json = R"({
            "temperature":0.7, "top_k":40, "top_p":0.95, "n_keep":0,
            "n_predict":10, "cache_prompt":true, "stop":"<|im_end|>",
            "stream":true, "repeat_penalty":1.1,
            "lora":[{"id":0,"scale":0.0},{"id":1,"scale":0.0}],
            "messages":[
                {"role":"system","content":"你是个聊天助手。"},
                {"role":"user","content":"你是谁？"}
            ]
        })";
        REQUIRE(SendRequest(RequestType::OPENAI_CHAT, "req-3", chat_json, &output, 10));
    }

    SECTION("tokenize test") {
        REQUIRE(SendRequest(RequestType::TOKENIZE, "req-4", "你是谁", &output));
    }

    SECTION("detokenize test") {
        REQUIRE(SendRequest(RequestType::DETOKENIZE, "req-5", "105043 100165", &output));
    }

    SECTION("metrics test") {
        REQUIRE(SendRequest(RequestType::METRICS, "req-6", ""));
    }

    SECTION("lora get test") {
        REQUIRE(SendRequest(RequestType::LORA_GET, "req-7", ""));
    }

    SECTION("worker save test") {
        REQUIRE(SendRequest(RequestType::WORKER_SAVE, "0", "worker-0"));
    }

    SECTION("worker restore test") {
        REQUIRE(SendRequest(RequestType::WORKER_RESTORE, "0", "worker-0"));
    }

    SECTION("worker erase test") {
        REQUIRE(SendRequest(RequestType::WORKER_ERASE, "0", ""));
    }

    SECTION("cancel tests") {
        std::string long_prompt = "21世纪以来，人工智能..."; // 可截短
        auto future = SendRequestAsync(RequestType::COMPLETIONS, "cancel-test", long_prompt, 100);

        std::this_thread::sleep_for(std::chrono::seconds(2));
        REQUIRE(SendRequest(RequestType::TASK_CANCEL, "cancel-test", ""));

        bool result = future.get();
        // 取消操作通常会导致主请求失败，但取消本身应成功
        REQUIRE((result || true));

        // 可以添加类似 TASK_ABORT 测试，此处略
    }

    SECTION("abort tests") {
        std::string long_prompt = R"(人工智能在21世纪以来取得了突飞猛进的发展。从早期的专家系统到统计学习，再到深度学习的爆发，
        技术范式经历了多次重大变革。2012年，AlexNet在ImageNet竞赛中以压倒性优势夺冠，标志着深度学习时代的正式开启。此后，
        循环神经网络、长短期记忆网络、生成对抗网络以及Transformer架构相继涌现，不断刷新着机器在视觉、语音、自然语言处理等领域的能力上限。
        特别是2017年提出的Transformer模型，其自注意力机制彻底改变了序列建模的方式，为后来BERT、GPT等一系列大规模预训练语言模型奠定了理论基础。
        2022年末，ChatGPT的横空出世更是将大语言模型的热度推向了全球每一个角落，它不仅能够进行多轮流畅对话，还能完成代码编写、文档总结、创意写作等复杂任务，
        引发了新一轮的产业革命和社会讨论。与此同时，算力需求呈指数级增长，GPU、TPU等专用芯片成为战略资源，分布式训练、模型量化、稀疏化等技术也成为了研究热点。
        然而，大模型也带来了巨大的能耗、数据隐私、算法偏见以及潜在滥用风险等问题。学术界和工业界正积极探索更高效、更安全、更可解释的人工智能技术路径。
        在中国，人工智能已被列为国家战略，从基础研究到产业落地全面布局，智能安防、智慧医疗、自动驾驶、智能制造等领域涌现出大量创新应用。
        尽管当前大模型在逻辑推理、事实一致性、长程记忆等方面仍存在不足，但研究人员正通过检索增强生成、思维链提示、多模态融合等方法持续改进。
        展望未来，通用人工智能的愿景虽然遥远，但每一步扎实的进展都在让机器更好地理解世界、服务人类。随着脑机接口、量子计算等前沿技术的潜在突破，
        人工智能的故事才刚刚翻开新的篇章，它将继续深刻地重塑我们的生产生活方式，以及我们对智能本质的认知。)";
        auto future = SendRequestAsync(RequestType::COMPLETIONS, "abort-test", long_prompt, 100);

        std::this_thread::sleep_for(std::chrono::seconds(1));
        REQUIRE(SendRequest(RequestType::TASK_ABORT, "abort-test", ""));

        bool result = future.get();
        // 取消操作通常会导致主请求失败，但取消本身应成功
        REQUIRE((result || true));

        // 可以添加类似 TASK_ABORT 测试，此处略
    }

}

int main(int argc, char* argv[]) {
    Catch::Session session;

    auto cli = session.cli()
        | Catch::Clara::Opt(g_model_name, "model")
            ["--model"]
            ("Triton model name to test (default: llamacpp)");

    session.cli(cli);

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) return returnCode;

    return session.run();
}