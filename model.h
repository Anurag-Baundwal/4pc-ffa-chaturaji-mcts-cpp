#pragma once
#include <string>
#include <vector>
#include <array>
#include "onnxruntime_cxx_api.h"
#include "types.h"

namespace chaturaji_cpp {

class Model {
public:
    // Constructor loads the ONNX file
    Model(const std::string& model_path);

    // Disable copying because Ort::Session is not copyable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Synchronous batched evaluation
    std::vector<EvaluationResult> evaluate_batch(const std::vector<EvaluationRequest>& requests);

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    // Input/Output names for the ONNX graph
    const char* input_name_ = "input";
    std::array<const char*, 2> output_names_ = {"policy", "value"};
};

} // namespace chaturaji_cpp