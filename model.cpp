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
    bool use_cuda_allocator = false;
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
        use_cuda_allocator = true;

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
    } catch (...) {
        std::cerr << "[C++] Model: CUDA defined but initialization failed (unknown error)." << std::endl;
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

    // Initialize IOBinding for optimized data transfer
    io_binding_ = std::make_unique<Ort::IoBinding>(session_);

    // Pre-allocate a reasonable initial capacity
    check_and_grow_buffers(256);
}

Model::~Model() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (allocator_) {
        if (planes_buffer_) allocator_->Free(planes_buffer_);
        if (scalars_buffer_) allocator_->Free(scalars_buffer_);
        if (policy_buffer_) allocator_->Free(policy_buffer_);
        if (value_buffer_) allocator_->Free(value_buffer_);
    }
}

void Model::check_and_grow_buffers(size_t batch_size) {
    if (batch_size <= buffer_capacity_) return;

    // Grow by 1.5x or fit to request, whichever is larger
    size_t new_capacity = std::max(batch_size, (buffer_capacity_ * 3) / 2);

    // Release old memory
    if (planes_buffer_) allocator_->Free(planes_buffer_);
    if (scalars_buffer_) allocator_->Free(scalars_buffer_);
    if (policy_buffer_) allocator_->Free(policy_buffer_);
    if (value_buffer_) allocator_->Free(value_buffer_);

    // Set to null so if Alloc fails, the destructor doesn't double-free
    planes_buffer_ = nullptr; 
    scalars_buffer_ = nullptr;
    policy_buffer_ = nullptr;
    value_buffer_ = nullptr;

    // Allocate
    planes_buffer_ = static_cast<float*>(allocator_->Alloc(new_capacity * NN_INPUT_PLANES_SIZE * sizeof(float)));
    scalars_buffer_ = static_cast<float*>(allocator_->Alloc(new_capacity * NN_INPUT_SCALARS * sizeof(float)));
    policy_buffer_ = static_cast<float*>(allocator_->Alloc(new_capacity * NN_POLICY_SIZE * sizeof(float)));
    value_buffer_ = static_cast<float*>(allocator_->Alloc(new_capacity * NN_VALUE_SIZE * sizeof(float)));

    buffer_capacity_ = new_capacity;
    last_batch_size_ = 0; // Force re-binding because the raw pointers just changed
}

std::vector<EvaluationResult> Model::evaluate_batch(const std::vector<EvaluationRequest>& requests) {
    if (requests.empty()) return {};

    // Thread safety: Lock mutex to protect shared IOBinding and persistent buffers
    std::lock_guard<std::mutex> lock(model_mutex_);

    size_t batch_size = requests.size();
    
    // 1. Ensure persistent buffers are large enough
    check_and_grow_buffers(batch_size);
    
    // 2. Flatten requests into contiguous persistent buffers
    for (size_t i = 0; i < batch_size; ++i) {
        std::memcpy(planes_buffer_ + (i * NN_INPUT_PLANES_SIZE), requests[i].input_planes->data(), NN_INPUT_PLANES_SIZE * sizeof(float));
        std::memcpy(scalars_buffer_ + (i * NN_INPUT_SCALARS), requests[i].input_scalars->data(), NN_INPUT_SCALARS * sizeof(float));
    }

    // 3. Ony Re-Bind if the batch size changed
    if (batch_size != last_batch_size_) {
        io_binding_->ClearBoundInputs();
        io_binding_->ClearBoundOutputs();

        std::array<int64_t, 4> planes_shape = { (int64_t)batch_size, NN_INPUT_PLANES, BOARD_DIM, BOARD_DIM };
        std::array<int64_t, 2> scalars_shape = { (int64_t)batch_size, NN_INPUT_SCALARS };
        std::array<int64_t, 2> policy_shape = { (int64_t)batch_size, NN_POLICY_SIZE };
        std::array<int64_t, 2> value_shape = { (int64_t)batch_size, NN_VALUE_SIZE };

        // Bind Inputs
        io_binding_->BindInput(input_names_[0], Ort::Value::CreateTensor<float>(memory_info_, planes_buffer_, batch_size * NN_INPUT_PLANES_SIZE, planes_shape.data(), planes_shape.size()));
        io_binding_->BindInput(input_names_[1], Ort::Value::CreateTensor<float>(memory_info_, scalars_buffer_, batch_size * NN_INPUT_SCALARS, scalars_shape.data(), scalars_shape.size()));

        // Bind Outputs
        io_binding_->BindOutput(output_names_[0], Ort::Value::CreateTensor<float>(memory_info_, policy_buffer_, batch_size * NN_POLICY_SIZE, policy_shape.data(), policy_shape.size()));
        io_binding_->BindOutput(output_names_[1], Ort::Value::CreateTensor<float>(memory_info_, value_buffer_, batch_size * NN_VALUE_SIZE, value_shape.data(), value_shape.size()));

        last_batch_size_ = batch_size;
    }

    // 3. Run Inference (Fastest possible path)
    session_.Run(Ort::RunOptions{nullptr}, *io_binding_);

    // 7. Extract Results
    std::vector<EvaluationResult> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        EvaluationResult res;
        res.request_id = requests[i].request_id;
        
        res.policy_logits = TensorPool::acquire_policy();
        res.value = TensorPool::acquire_value();

        // Copy data from persistent buffers to individual result structures
        std::memcpy(res.policy_logits->data(), policy_buffer_ + (i * NN_POLICY_SIZE), NN_POLICY_SIZE * sizeof(float));
        std::memcpy(res.value->data(), value_buffer_ + (i * NN_VALUE_SIZE), NN_VALUE_SIZE * sizeof(float));
        
        results.push_back(std::move(res));
    }

    return results;
}

} // namespace chaturaji_cpp