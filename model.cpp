#include "model.h"
#include "onnxruntime_cxx_api.h" 
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <filesystem>
#include <cstring> // For std::memcpy

namespace fs = std::filesystem;
namespace chaturaji_cpp {

Model::Model(const std::string& model_path) :
    env_(ORT_LOGGING_LEVEL_WARNING, "ChaturajiInference"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
    session_(nullptr),
    allocator_(nullptr) 
{
    Ort::SessionOptions session_options;
    
    // Disable per-op threads to allow high throughput for simultaneous batches
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool provider_loaded = false;
    std::string final_model_path = model_path;

    // --- 1. Attempt to use CUDA (NVIDIA GPU) ---
#ifdef USE_CUDA
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0; 
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(cuda_options);
        
        // Enabled CUDA Host (Pinned) memory for faster DMA transfers
        memory_info_ = Ort::MemoryInfo("CudaHost", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU);

        std::cout << "[C++] Model: Enabled CUDA Execution Provider." << std::endl;
        provider_loaded = true;

        // --- FP16 Auto-Detection Logic ---
        if (model_path.size() > 5 && model_path.substr(model_path.size() - 5) == ".onnx") {
            std::string fp16_path = model_path.substr(0, model_path.size() - 5) + "_fp16.onnx";
            if (fs::exists(fp16_path)) {
                final_model_path = fp16_path;
                std::cout << "[C++] Model: Found and using FP16 model for GPU: " << final_model_path << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[C++] Model: CUDA defined but initialization failed: " << e.what() << std::endl;
    }
#endif

    // --- 2. Attempt OpenVINO (Intel iGPU/CPU) ---
    if (!provider_loaded) {
        try {
            std::unordered_map<std::string, std::string> ov_options;
            ov_options["device_type"] = "GPU"; 
            session_options.AppendExecutionProvider("OpenVINO", ov_options);
            
            std::cout << "[C++] Model: Enabled OpenVINO Execution Provider." << std::endl;
            provider_loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "[C++] Model: Warning: OpenVINO setup failed: " << e.what() << std::endl;
        }
    }

    // --- 3. CPU Fallback ---
    if (!provider_loaded) {
        std::cout << "[C++] Model: Falling back to CPU execution." << std::endl;
    }

    // Load the session using the selected path (FP16 or Standard)
    std::filesystem::path p(final_model_path);
#ifdef _WIN32
    session_ = Ort::Session(env_, p.wstring().c_str(), session_options);
#else
    session_ = Ort::Session(env_, p.c_str(), session_options);
#endif

    // Initialize the allocator pointer
    allocator_ = std::make_unique<Ort::Allocator>(session_, memory_info_);

    // Initialize Contexts
    for(auto& ctx : contexts_) {
        ctx.io_binding = std::make_unique<Ort::IoBinding>(session_);
        check_and_grow_context(ctx, 256); // Pre-allocate initial capacity
    }
}

Model::~Model() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (allocator_) {
        for(auto& ctx : contexts_) {
            if (ctx.planes_buffer) allocator_->Free(ctx.planes_buffer);
            if (ctx.scalars_buffer) allocator_->Free(ctx.scalars_buffer);
            if (ctx.policy_buffer) allocator_->Free(ctx.policy_buffer);
            if (ctx.value_buffer) allocator_->Free(ctx.value_buffer);
        }
    }
}

void Model::check_and_grow_context(BatchContext& ctx, size_t batch_size) {
    if (batch_size <= ctx.buffer_capacity) return;

    size_t new_capacity = std::max(batch_size, (ctx.buffer_capacity * 3) / 2);

    // Release old memory
    if (ctx.planes_buffer) allocator_->Free(ctx.planes_buffer);
    if (ctx.scalars_buffer) allocator_->Free(ctx.scalars_buffer);
    if (ctx.policy_buffer) allocator_->Free(ctx.policy_buffer);
    if (ctx.value_buffer) allocator_->Free(ctx.value_buffer);

    // Allocate using ORT Allocator (Pinned/CudaHost if enabled)
    ctx.planes_buffer = static_cast<float*>(allocator_->Alloc(new_capacity * NN_INPUT_PLANES_SIZE * sizeof(float)));
    ctx.scalars_buffer = static_cast<float*>(allocator_->Alloc(new_capacity * NN_INPUT_SCALARS * sizeof(float)));
    ctx.policy_buffer = static_cast<float*>(allocator_->Alloc(new_capacity * NN_POLICY_SIZE * sizeof(float)));
    ctx.value_buffer = static_cast<float*>(allocator_->Alloc(new_capacity * NN_VALUE_SIZE * sizeof(float)));

    ctx.buffer_capacity = new_capacity;

    // CRITICAL: Force re-binding because raw pointers changed
    ctx.last_batch_size = 0; 
    ctx.planes_val = Ort::Value{nullptr};
    ctx.scalars_val = Ort::Value{nullptr};
    ctx.policy_val = Ort::Value{nullptr};
    ctx.value_val = Ort::Value{nullptr};
}

std::vector<EvaluationResult> Model::evaluate_batch(std::vector<EvaluationRequest>& requests) {
    // Pass '0' as the context index for fallback synchronous calls
    return evaluate_batch_async(requests, 0).get();
}

std::future<std::vector<EvaluationResult>> Model::evaluate_batch_async(std::vector<EvaluationRequest>& requests, int context_idx) {
    if (requests.empty()) {
        std::promise<std::vector<EvaluationResult>> p;
        p.set_value({});
        return p.get_future();
    }

    // Lock protects the shared context array and buffer growth logic
    std::lock_guard<std::mutex> lock(model_mutex_);

    BatchContext& ctx = contexts_[context_idx];
    size_t batch_size = requests.size();
    check_and_grow_context(ctx, batch_size);

    // CPU WORK: Synchronous copy to pinned memory
    std::vector<RequestId> req_ids;
    req_ids.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        req_ids.push_back(requests[i].request_id);
        
        std::memcpy(ctx.planes_buffer + (i * NN_INPUT_PLANES_SIZE), 
                    requests[i].input_planes->data(), NN_INPUT_PLANES_SIZE * sizeof(float));
        std::memcpy(ctx.scalars_buffer + (i * NN_INPUT_SCALARS), 
                    requests[i].input_scalars->data(), NN_INPUT_SCALARS * sizeof(float));
        
        // Return memory to pool immediately so MCTS workers can reuse it
        TensorPool::release_planes(std::move(requests[i].input_planes));
        TensorPool::release_scalars(std::move(requests[i].input_scalars));
    }

    // Launch Inference in background
    return std::async(std::launch::async, [this, &ctx, batch_size, ids = std::move(req_ids)]() mutable {
        
        // Persistent Binding Logic: Only re-bind if the batch size changed.
        // The handles are stored in 'ctx' so they persist between calls.
        if (batch_size != ctx.last_batch_size) {
            ctx.io_binding->ClearBoundInputs();
            ctx.io_binding->ClearBoundOutputs();

            std::array<int64_t, 4> planes_shape = { (int64_t)batch_size, NN_INPUT_PLANES, BOARD_DIM, BOARD_DIM };
            std::array<int64_t, 2> scalars_shape = { (int64_t)batch_size, NN_INPUT_SCALARS };
            std::array<int64_t, 2> policy_shape = { (int64_t)batch_size, NN_POLICY_SIZE };
            std::array<int64_t, 2> value_shape = { (int64_t)batch_size, NN_VALUE_SIZE };

            // 1. Create and store persistent handles
            ctx.planes_val = Ort::Value::CreateTensor<float>(memory_info_, ctx.planes_buffer, batch_size * NN_INPUT_PLANES_SIZE, planes_shape.data(), planes_shape.size());
            ctx.scalars_val = Ort::Value::CreateTensor<float>(memory_info_, ctx.scalars_buffer, batch_size * NN_INPUT_SCALARS, scalars_shape.data(), scalars_shape.size());
            ctx.policy_val = Ort::Value::CreateTensor<float>(memory_info_, ctx.policy_buffer, batch_size * NN_POLICY_SIZE, policy_shape.data(), policy_shape.size());
            ctx.value_val = Ort::Value::CreateTensor<float>(memory_info_, ctx.value_buffer, batch_size * NN_VALUE_SIZE, value_shape.data(), value_shape.size());

            // 2. Bind the handles to the session
            ctx.io_binding->BindInput(input_names_[0], ctx.planes_val);
            ctx.io_binding->BindInput(input_names_[1], ctx.scalars_val);
            ctx.io_binding->BindOutput(output_names_[0], ctx.policy_val);
            ctx.io_binding->BindOutput(output_names_[1], ctx.value_val);

            ctx.last_batch_size = batch_size;
        }

        // 3. Execute Inference (Fast path)
        session_.Run(Ort::RunOptions{nullptr}, *ctx.io_binding);

        // 4. Package Results
        std::vector<EvaluationResult> results;
        results.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            EvaluationResult res;
            res.request_id = ids[i];
            res.policy_logits = TensorPool::acquire_policy();
            res.value = TensorPool::acquire_value();
            std::memcpy(res.policy_logits->data(), ctx.policy_buffer + (i * NN_POLICY_SIZE), NN_POLICY_SIZE * sizeof(float));
            std::memcpy(res.value->data(), ctx.value_buffer + (i * NN_VALUE_SIZE), NN_VALUE_SIZE * sizeof(float));
            results.push_back(std::move(res));
        }
        return results;
    });
}

} // namespace chaturaji_cpp