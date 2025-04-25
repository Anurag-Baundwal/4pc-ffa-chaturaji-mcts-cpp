# BUILD (or BUILD.bazel)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

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
# This part remains largely the same. The actual libtorch target they
# point to will be defined in the WORKSPACE file, which is platform-specific.
# On Windows, @libtorch_release and @libtorch_debug point to Windows CPU libs.
# On Colab, the notebook script modifies WORKSPACE so @libtorch_release
# points to the Linux GPU lib.
cc_library(
    name = "libtorch_configured",
    deps = select({
        # Use debug libtorch on Windows debug builds
        "//:debug_build": ["@libtorch_debug//:libtorch_debug"],
        # Use release libtorch otherwise (Windows release or Colab GPU/CPU)
        "//conditions:default": ["@libtorch_release//:libtorch"],
    }),
)


# === All cc_library and cc_binary rules need their copts updated ===

cc_library(
    name = "chaturaji_types",
    hdrs = [
        "types.h",
        "piece.h",
    ],
    copts = select({
        # If building with --define use_cuda=true (Colab GPU likely)
        "//:cuda_build": [
            "-std=c++17",         # GCC/Clang C++17 flag
            "-D AT_CUDA_ENABLED=1", # Define the CUDA macro
        ],
        # Default case (Windows MSVC, potentially other non-CUDA Linux)
        # This covers both Windows Debug and Release builds.
        "//conditions:default": [
            "/std:c++17", # MSVC C++17 flag
            # Add other general compiler flags here if needed, e.g., optimization flags like /O2
        ],
    }),
)

cc_library(
    name = "chaturaji_board",
    srcs = ["board.cpp"],
    hdrs = ["board.h"],
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
    ],
)

# Repeat the 'copts = select({...})' block for all remaining rules:
# chaturaji_utils, chaturaji_model, chaturaji_mcts_node,
# chaturaji_search, chaturaji_self_play, chaturaji_training,
# chaturaji_engine, zobrist_test.

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
        ":libtorch_configured",
    ],
)

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

cc_binary(
    name = "chaturaji_engine",
    srcs = ["main.cpp"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_board",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_training",
        ":chaturaji_types",
        ":chaturaji_utils",
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
    ],
    linkstatic = True,
)