# BUILD (or BUILD.bazel)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")

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
        "thread_safe_queue.h",
    ],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":libtorch_configured",
    ],
)

# === Magic Utilities Library ===
cc_library(
    name = "chaturaji_magic_utils",
    srcs = ["magic_utils.cpp"],
    hdrs = ["magic_utils.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_types",
    ],
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
        ":chaturaji_types",
        ":chaturaji_magic_utils",
    ],
)

cc_library(
    name = "chaturaji_utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h", "data_writer.h"], # Added data_writer.h here
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
        ":chaturaji_magic_utils",
        ":libtorch_configured",
    ],
)

# === MCTSNode Forward Declaration Library ===
# This library just contains the forward declaration header for MCTSNode.
cc_library(
    name = "chaturaji_mcts_node_fwd",
    hdrs = ["mcts_node_fwd.h"],
)

# === Node Pool Library ===
# Manages memory allocation for MCTSNode objects.
cc_library(
    name = "chaturaji_mcts_node_pool",
    srcs = ["mcts_node_pool.cpp"],
    hdrs = ["mcts_node_pool.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_mcts_node_fwd", # mcts_node_pool.h needs the forward declaration
          ":chaturaji_types",         # For std::mutex, etc. (and transitive libtorch if needed)
          # REMOVED: ":chaturaji_mcts_node", # This was the direct cause of the cycle!
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
        ":chaturaji_types",
        # thread_safe_queue.h is declared in chaturaji_types's hdrs, so this dependency is sufficient.
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
        ":chaturaji_types",
        ":chaturaji_mcts_node_pool", # Direct dependency on the node pool library
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
        ":chaturaji_model",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured",
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
        ":chaturaji_evaluator",
        ":chaturaji_mcts_node",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured",
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
        ":chaturaji_self_play",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured",
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
        ":chaturaji_board",
        ":chaturaji_evaluator",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_self_play",
        ":chaturaji_strength_test",
        ":chaturaji_training",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":chaturaji_magic_utils",
        ":libtorch_configured",
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
        ":chaturaji_types",
        ":chaturaji_mcts_node", # Zobrist test doesn't directly use MCTSNode, but if it runs through main.cpp
                               # or initializes a board, it might pull in related dependencies.
                               # It's better to ensure its dependencies are complete.
    ],
)

cc_library(
    name = "chaturaji_strength_test",
    srcs = ["strength_test.cpp"],
    hdrs = ["strength_test.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured",
    ],
)

cc_binary(
    name = "magic_finder",
    srcs = ["magic_finder.cpp"],
    copts = select({
        "//:cuda_build": [
            "-std=c++17",
            "-D AT_CUDA_ENABLED=1",
        ],
        "//conditions:default": [
            "/std:c++17",
        ],
    }),
    deps = [
        ":chaturaji_types",
        ":chaturaji_magic_utils",
    ],
)

# === Perft ===

cc_binary(
    name = "perft",
    srcs = ["perft.cpp"],
    copts = select({
        "//:cuda_build": [
            "-std=c++17",
            "-D AT_CUDA_ENABLED=1",
        ],
        "//conditions:default": [
            "/std:c++17",
        ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_utils",
        ":chaturaji_types",
        ":chaturaji_magic_utils",
        # Note: We generally don't need the neural network libs for perft
        # unless types.h or board.h implicitly pulls them in via includes.
        # Based on your files, board.h pulls types.h which pulls libtorch.
        # So we might need libtorch here solely for linking purposes if headers use it.
        ":libtorch_configured", 
    ],
)