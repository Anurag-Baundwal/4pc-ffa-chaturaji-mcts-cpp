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
    name = "chaturaji_magic_utils",  # Consistent naming convention
    srcs = ["magic_utils.cpp"],
    hdrs = ["magic_utils.h"],
    copts = select({
        "//:cuda_build": [ "-std=c++17", "-D AT_CUDA_ENABLED=1", ],
        "//conditions:default": [ "/std:c++17", ],
    }),
    deps = [
        ":chaturaji_types", # magic_utils.h includes types.h
                           # No need for libtorch_configured directly here if types.h handles it
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
        ":chaturaji_magic_utils", # board.h and board.cpp use magic_utils
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
        ":chaturaji_magic_utils", # utils.cpp uses magic_utils
        ":libtorch_configured",
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
        ":chaturaji_board", # mcts_node.cpp includes board.h which now includes magic_utils.h
        ":chaturaji_types",
        # ":chaturaji_magic_utils", # Implicitly included via chaturaji_board
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
        ":chaturaji_utils", # utils.h might be included directly or transitively.
                            # search.h includes utils.h.
        # ":chaturaji_magic_utils", # Implicit via board or utils
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
        # ":chaturaji_magic_utils", # Implicit
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
        # ":chaturaji_magic_utils", # Implicit
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
        ":chaturaji_magic_utils", # main.cpp might not directly use it, but good to link all our libs
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
        ":chaturaji_board", # board.h includes magic_utils.h
        ":chaturaji_types",
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
        # ":chaturaji_magic_utils", # Implicit
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
        ":chaturaji_magic_utils", # magic_finder.cpp directly includes magic_utils.h
    ],
)