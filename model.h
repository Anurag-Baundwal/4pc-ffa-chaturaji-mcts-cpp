#pragma once
#include <string>
#include <vector>
#include <array>
#include <memory> 
#include <mutex>  
#include <future> 
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

    // Synchronous batched evaluation (Legacy / Fallback)
    std::vector<EvaluationResult> evaluate_batch(std::vector<EvaluationRequest>& requests);

    // Asynchronous batched evaluation (Pipelined)
    // Returns a future that resolves to the results.
    // NOTE: This performs the CPU memory copy synchronously, then launches the GPU inference asynchronously.
    std::future<std::vector<EvaluationResult>> evaluate_batch_async(std::vector<EvaluationRequest>& requests, int context_idx);

private:
    // Structure to hold state for a single pipeline stage (Triple Buffering)
    struct BatchContext {
        std::unique_ptr<Ort::IoBinding> io_binding;
        
        // Raw pointers for persistent buffers (pinned memory)
        float* planes_buffer = nullptr;
        float* scalars_buffer = nullptr;
        float* policy_buffer = nullptr;
        float* value_buffer = nullptr;
        
        size_t buffer_capacity = 0;
        size_t last_batch_size = 0;

        // Persistent handles to the tensors. 
        // These must stay alive as long as the io_binding is using them.
        Ort::Value planes_val{nullptr};
        Ort::Value scalars_val{nullptr};
        Ort::Value policy_val{nullptr};
        Ort::Value value_val{nullptr};
    };

    // Helper to resize persistent buffers for a specific context
    void check_and_grow_context(BatchContext& ctx, size_t batch_size);

    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    // --- Optimization: Triple-Buffered Pipeline ---
    // We use three contexts to maximize device saturation. While the GPU is executing 
    // Context A, the Evaluator can be processing the completed results of Context B 
    // while simultaneously prepping the next batch in Context C.
    static constexpr int NUM_CONTEXTS = 3;
    std::array<BatchContext, NUM_CONTEXTS> contexts_;

    // Protects access to the context selection and buffer growth logic
    std::mutex model_mutex_;

    // Aligned memory allocator (Pointer used to handle runtime initialization)
    std::unique_ptr<Ort::Allocator> allocator_;

    // Input/Output names for the ONNX graph
    std::array<const char*, 2> input_names_ = {"input_planes", "input_scalars"};
    std::array<const char*, 2> output_names_ = {"policy", "value"};
};

} // namespace chaturaji_cpp