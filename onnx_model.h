#pragma once
#include <string>
#include <vector>
#include <array>
#include "onnxruntime_cxx_api.h"
#include "types.h"

namespace chaturaji_cpp {

class OnnxModel {
public:
    OnnxModel(const std::string& model_path);

    // Synchronous batched evaluation
    std::vector<EvaluationResult> evaluate_batch(const std::vector<EvaluationRequest>& requests);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    // Input/Output names
    const char* input_name_ = "input";
    std::array<const char*, 2> output_names_ = {"policy", "value"};
};

}