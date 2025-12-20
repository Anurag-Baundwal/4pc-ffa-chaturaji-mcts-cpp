#include "onnx_model.h"
#include <algorithm>
#include <stdexcept>

namespace chaturaji_cpp {

OnnxModel::OnnxModel(const std::string& model_path) :
    env_(ORT_LOGGING_LEVEL_WARNING, "ChaturajiInference"),
    session_(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), Ort::SessionOptions{nullptr}),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) 
{
    // Optional: Set thread count for ORT
    // Ort::SessionOptions options;
    // options.SetIntraOpNumThreads(4); 
}

std::vector<EvaluationResult> OnnxModel::evaluate_batch(const std::vector<EvaluationRequest>& requests) {
    if (requests.empty()) return {};

    size_t batch_size = requests.size();
    size_t input_count = 33 * 8 * 8;
    
    // 1. Flatten all requests into one contiguous buffer
    std::vector<float> input_tensor_values(batch_size * input_count);
    for (size_t i = 0; i < batch_size; ++i) {
        std::copy(requests[i].state_floats.begin(), requests[i].state_floats.end(), 
                  input_tensor_values.begin() + (i * input_count));
    }

    // 2. Wrap buffer in ORT Tensor
    std::array<int64_t, 4> input_shape = { (int64_t)batch_size, 33, 8, 8 };
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
        
        // Copy policy (4096)
        std::copy(policy_ptr + (i * 4096), policy_ptr + ((i + 1) * 4096), res.policy_logits.begin());
        
        // Copy values (4)
        std::copy(value_ptr + (i * 4), value_ptr + ((i + 1) * 4), res.value.begin());
        
        results.push_back(res);
    }

    return results;
}

}