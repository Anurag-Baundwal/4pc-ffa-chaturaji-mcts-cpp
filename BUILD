load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")

package(default_visibility = ["//visibility:public"])

# Settings for different build modes
config_setting(
    name = "debug_build",
    values = {"compilation_mode": "dbg"},
)

config_setting(
    name = "cuda_build",
    define_values = {"use_cuda": "true"},
)

# === Base Types and Utilities ===

cc_library(
    name = "chaturaji_types",
    hdrs = [
        "types.h",
        "piece.h",
        "thread_safe_queue.h",
    ],
    copts = ["/std:c++17"],
    deps = [], # Libtorch removed
)

cc_library(
    name = "chaturaji_magic_utils",
    srcs = ["magic_utils.cpp"],
    hdrs = ["magic_utils.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_types",
    ],
)

cc_library(
    name = "chaturaji_board",
    srcs = ["board.cpp"],
    hdrs = ["board.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_types",
        ":chaturaji_magic_utils",
    ],
)

cc_library(
    name = "chaturaji_utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h", "data_writer.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
        ":chaturaji_magic_utils",
        # Libtorch removed
    ],
)

# === MCTSNode and Memory Management ===

cc_library(
    name = "chaturaji_mcts_node_fwd",
    hdrs = ["mcts_node_fwd.h"],
)

cc_library(
    name = "chaturaji_mcts_node_pool",
    srcs = ["mcts_node_pool.cpp"],
    hdrs = ["mcts_node_pool.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_mcts_node_fwd",
        ":chaturaji_types",
    ],
)

# === Core ML and Search Components ===

cc_library(
    name = "chaturaji_model",
    srcs = ["model.cpp"],
    hdrs = ["model.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_types",
        "@onnxruntime//:onnxruntime", # Switched to ONNX
    ],
)

cc_library(
    name = "chaturaji_evaluator",
    srcs = ["evaluator.cpp"],
    hdrs = ["evaluator.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_model",
        ":chaturaji_types",
    ],
)

cc_library(
    name = "chaturaji_mcts_node",
    srcs = ["mcts_node.cpp"],
    hdrs = ["mcts_node.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
        ":chaturaji_mcts_node_pool",
    ],
)

cc_library(
    name = "chaturaji_search",
    srcs = ["search.cpp"],
    hdrs = ["search.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_mcts_node",
        ":chaturaji_model",
        ":chaturaji_types",
        ":chaturaji_utils",
    ],
)

# === Self-Play and Training ===

cc_library(
    name = "chaturaji_self_play",
    srcs = ["self_play.cpp"],
    hdrs = ["self_play.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_evaluator",
        ":chaturaji_mcts_node",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_types",
        ":chaturaji_utils",
    ],
)

cc_library(
    name = "chaturaji_training",
    srcs = ["train.cpp"],
    hdrs = ["train.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_model",
        ":chaturaji_self_play",
        ":chaturaji_types",
        ":chaturaji_utils",
    ],
)

cc_library(
    name = "chaturaji_strength_test",
    srcs = ["strength_test.cpp"],
    hdrs = ["strength_test.h"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_types",
        ":chaturaji_utils",
    ],
)

# === Executables and Tests ===

cc_binary(
    name = "chaturaji_engine",
    srcs = ["main.cpp"],
    copts = ["/std:c++17"],
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
    ],
)

cc_test(
    name = "zobrist_test",
    srcs = ["zobrist_test.cpp"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
        ":chaturaji_mcts_node",
    ],
)

cc_binary(
    name = "magic_finder",
    srcs = ["magic_finder.cpp"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_types",
        ":chaturaji_magic_utils",
    ],
)

cc_binary(
    name = "perft",
    srcs = ["perft.cpp"],
    copts = ["/std:c++17"],
    deps = [
        ":chaturaji_board",
        ":chaturaji_utils",
        ":chaturaji_types",
        ":chaturaji_magic_utils",
    ],
)