# BUILD (or BUILD.bazel)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test") # Added cc_test

package(default_visibility = ["//visibility:public"])

# Define a configuration setting for debug builds (still useful for Windows)
config_setting(
    name = "debug_build",
    values = {"compilation_mode": "dbg"},
)

# Define a configuration setting for CUDA builds (trigger with --define use_cuda=true)
config_setting(
    name = "cuda_build",
    define_values = {"use_cuda": "true"},
)

# --- Intermediate library to handle libtorch selection ---
cc_library(
    name = "libtorch_configured",
    deps = select({
        "//:debug_build": ["@libtorch_debug//:libtorch_debug"],
        "//conditions:default": ["@libtorch_release//:libtorch"],
    }),
)


# === Base Types and Utilities ===

cc_library(
    name = "chaturaji_types",
    hdrs = [
        "types.h",
        "piece.h",
        "thread_safe_queue.h", # ADDED thread_safe_queue header
    ],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    # --- ADDED DEPENDENCY HERE ---
    deps = [
        ":libtorch_configured", # Needed because types.h now includes torch/torch.h
    ],
    # --- END ADDED DEPENDENCY ---
)

cc_library(
    name = "chaturaji_board",
    srcs = ["board.cpp"],
    hdrs = ["board.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_types", # This now transitively provides libtorch includes
    ],
)

cc_library(
    name = "chaturaji_utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
     copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
        ":libtorch_configured", # Keep explicit dependency for clarity too
    ],
)

# === Core ML and Search Components ===

cc_library(
    name = "chaturaji_model",
    srcs = ["model.cpp"],
    hdrs = ["model.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_types",
        ":libtorch_configured",
    ],
)

# NEW: Evaluator component
cc_library(
    name = "chaturaji_evaluator",
    srcs = ["evaluator.cpp"],
    hdrs = ["evaluator.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_model",
        ":chaturaji_types", # Depends on EvaluationRequest/Result, ThreadSafeQueue
        ":libtorch_configured",
    ],
)

cc_library(
    name = "chaturaji_mcts_node",
    srcs = ["mcts_node.cpp"],
    hdrs = ["mcts_node.h"],
     copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_types", # This now transitively provides libtorch includes
    ],
)

cc_library(
    name = "chaturaji_search",
    srcs = ["search.cpp"],
    hdrs = ["search.h"],
     copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_mcts_node",
        ":chaturaji_model", # Still needed for sync mode
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Still needed for sync mode
    ],
)

# === Self-Play and Training ===

cc_library(
    name = "chaturaji_self_play",
    srcs = ["self_play.cpp"],
    hdrs = ["self_play.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_evaluator", # Depends on Evaluator now
        ":chaturaji_mcts_node",
        ":chaturaji_model",     # Still needed to pass handle to Evaluator
        ":chaturaji_search",    # Needs get_reward_map, process_policy etc.
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Keep explicit dependency
    ],
)

cc_library(
    name = "chaturaji_training",
    srcs = ["train.cpp"],
    hdrs = ["train.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_model",
        ":chaturaji_self_play", # Depends on SelfPlay
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Keep explicit dependency
    ],
)

# === Executable and Tests ===

cc_binary(
    name = "chaturaji_engine",
    srcs = ["main.cpp"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        # Include all necessary components
        ":chaturaji_board",
        ":chaturaji_evaluator", # Needed by SelfPlay
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_self_play", # Needed by Training
        ":chaturaji_strength_test", # NEW: Strength test component
        ":chaturaji_training",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Keep explicit dependency
    ],
)

cc_test(
    name = "zobrist_test",
    srcs = ["zobrist_test.cpp"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_types", # This now transitively provides libtorch includes
        # Test doesn't need libtorch linked explicitly, just headers via types
    ],
    # linkstatic = True, # Consider removing if causing issues with DLLs/shared libs
)

# === NEW: Strength Test ===
cc_library(
    name = "chaturaji_strength_test",
    srcs = ["strength_test.cpp"],
    hdrs = ["strength_test.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        # Needs components used for running games synchronously
        ":chaturaji_board",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured",
    ],
)