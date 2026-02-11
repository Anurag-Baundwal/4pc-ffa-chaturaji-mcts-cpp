#pragma once
#include <string>
#include <vector>
#include <array>
#include <memory> // Added for std::unique_ptr
#include <mutex>  // Added for std::mutex
#include "onnxruntime_cxx_api.h"
#include "types.h"
#include "utils.h"

namespace chaturaji_cpp {

class Model {
public:
    // Constructor loads the ONNX file
    Model(const std::string& model_path);

    // Destructor required to free raw allocated buffers
    ~Model();

    // Disable copying because Ort::Session is not copyable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Synchronous batched evaluation
    std::vector<EvaluationResult> evaluate_batch(const std::vector<EvaluationRequest>& requests);

private:
    // Helper to resize persistent buffers if the batch size exceeds current capacity
    void check_and_grow_buffers(size_t batch_size);

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    // --- Optimization: Persistent Buffers & IoBinding ---
    // Protects the shared stateful buffers (io_binding_, planes_buffer_, etc.) 
    // to allow safe concurrent access if the Model instance is shared.
    std::mutex model_mutex_;

    // IOBinding interface to avoid intermediate copies within ORT
    std::unique_ptr<Ort::IoBinding> io_binding_;

    // Aligned memory allocator (Pointer used to handle runtime initialization)
    std::unique_ptr<Ort::Allocator> allocator_;

    // Raw pointers for persistent buffers
    float* planes_buffer_ = nullptr;
    float* scalars_buffer_ = nullptr;
    float* policy_buffer_ = nullptr;
    float* value_buffer_ = nullptr;
    size_t buffer_capacity_ = 0;
    size_t last_batch_size_ = 0;

    // Input/Output names for the ONNX graph
    std::array<const char*, 2> input_names_ = {"input_planes", "input_scalars"};
    std::array<const char*, 2> output_names_ = {"policy", "value"};
};

} // namespace chaturaji_cpp