#include "model.h"
#include "onnxruntime_cxx_api.h" 
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;
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
            
            std::cout << "[C++] Model: Enabled CUDA Execution Provider." << std::endl;
            provider_loaded = true;

            // --- FP16 Auto-Detection Logic ---
            // If we are on CUDA, check if an FP16 version of the model exists.
            // Convention: "model.onnx" -> "model_fp16.onnx"
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
    session_ = Ort::Session(env_, std::wstring(final_model_path.begin(), final_model_path.end()).c_str(), session_options);
}

std::vector<EvaluationResult> Model::evaluate_batch(const std::vector<EvaluationRequest>& requests) {
    if (requests.empty()) return {};

    size_t batch_size = requests.size();
    
    // 1. Flatten requests into contiguous buffers
    std::vector<float> planes_buffer(batch_size * NN_INPUT_PLANES_SIZE);
    std::vector<float> scalars_buffer(batch_size * NN_INPUT_SCALARS);
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Copy Planes
        std::copy(requests[i].input_planes.begin(), requests[i].input_planes.end(), 
                  planes_buffer.begin() + (i * NN_INPUT_PLANES_SIZE));
        
        // Copy Scalars
        std::copy(requests[i].input_scalars.begin(), requests[i].input_scalars.end(),
                  scalars_buffer.begin() + (i * NN_INPUT_SCALARS));
    }

    // 2. Create ORT Tensors
    std::array<int64_t, 4> planes_shape = { (int64_t)batch_size, NN_INPUT_PLANES, BOARD_DIM, BOARD_DIM };
    std::array<int64_t, 2> scalars_shape = { (int64_t)batch_size, NN_INPUT_SCALARS };
    
    Ort::Value planes_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, planes_buffer.data(), planes_buffer.size(), 
        planes_shape.data(), planes_shape.size());

    Ort::Value scalars_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, scalars_buffer.data(), scalars_buffer.size(), 
        scalars_shape.data(), scalars_shape.size());

    // 3. Run Inference (with 2 inputs)
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(planes_tensor));
    input_tensors.push_back(std::move(scalars_tensor));

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), input_tensors.data(), 2, 
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