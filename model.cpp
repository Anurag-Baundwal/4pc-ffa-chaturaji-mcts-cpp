#include "model.h"
#include "onnxruntime_cxx_api.h" 
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

namespace chaturaji_cpp {

Model::Model(const std::string& model_path) :
    env_(ORT_LOGGING_LEVEL_WARNING, "ChaturajiInference"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
    session_(nullptr) 
{
    Ort::SessionOptions session_options;
    
    // Disable per-op threads to allow high throughput for simultaneous batches
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool provider_loaded = false;

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
            
            std::cout << "[C++] Model: Enabled CUDA Execution Provider." << std::endl;
            provider_loaded = true;
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

    session_ = Ort::Session(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
}

std::vector<EvaluationResult> Model::evaluate_batch(const std::vector<EvaluationRequest>& requests) {
    if (requests.empty()) return {};

    size_t batch_size = requests.size();
    
    // 1. Flatten all requests into one contiguous buffer
    std::vector<float> input_tensor_values(batch_size * NN_INPUT_SIZE);
    
    for (size_t i = 0; i < batch_size; ++i) {
        if (requests[i].state_floats.size() != NN_INPUT_SIZE) {
             throw std::runtime_error("Input state float size mismatch in Model::evaluate_batch");
        }
        std::copy(requests[i].state_floats.begin(), requests[i].state_floats.end(), 
                  input_tensor_values.begin() + (i * NN_INPUT_SIZE));
    }

    // 2. Wrap buffer in ORT Tensor
    // Shape: [Batch, 37, 8, 8]
    std::array<int64_t, 4> input_shape = { (int64_t)batch_size, NN_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM };
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_values.size(), 
        input_shape.data(), input_shape.size());

    // 3. Run Inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        &input_name_, &input_tensor, 1, 
        output_names_.data(), output_names_.size());

    // 4. Extract Results
    float* policy_ptr = output_tensors[0].GetTensorMutableData<float>();
    float* value_ptr = output_tensors[1].GetTensorMutableData<float>();

    std::vector<EvaluationResult> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        EvaluationResult res;
        res.request_id = requests[i].request_id;
        
        // Copy policy (4096 floats)
        std::copy(policy_ptr + (i * NN_POLICY_SIZE), policy_ptr + ((i + 1) * NN_POLICY_SIZE), res.policy_logits.begin());
        
        // Copy values (4 floats)
        std::copy(value_ptr + (i * NN_VALUE_SIZE), value_ptr + ((i + 1) * NN_VALUE_SIZE), res.value.begin());
        
        results.push_back(res);
    }

    return results;
}

} // namespace chaturaji_cpp