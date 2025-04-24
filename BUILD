# BUILD (or BUILD.bazel)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Define a configuration setting for debug builds
config_setting(
    name = "debug_build",
    values = {"compilation_mode": "dbg"},
)

# --- Intermediate library to handle libtorch selection ---
cc_library(
    name = "libtorch_configured",
    deps = select({
        "//:debug_build": ["@libtorch_debug//:libtorch_debug"],
        "//conditions:default": ["@libtorch_release//:libtorch"],
    }),
)

# Common C++17 compiler options for MSVC
MSVC_CXX17_COPTS = ["/std:c++17"]

# === Phase 1 Libraries ===
cc_library(
    name = "chaturaji_types",
    hdrs = [
        "types.h",
        "piece.h",
    ],
)

cc_library(
    name = "chaturaji_board",
    srcs = ["board.cpp"],
    hdrs = ["board.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_types",
    ],
)

# === Phase 2 Libraries ===

cc_library(
    name = "chaturaji_utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
        ":libtorch_configured", # Depends on the selector library
    ],
)

cc_library(
    name = "chaturaji_model",
    srcs = ["model.cpp"],
    hdrs = ["model.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_types",
        ":libtorch_configured", # Depends on the selector library
    ],
)

cc_library(
    name = "chaturaji_mcts_node",
    srcs = ["mcts_node.cpp"],
    hdrs = ["mcts_node.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_board",
        ":chaturaji_types",
    ],
)

# === Phase 3 Libraries ===

cc_library(
    name = "chaturaji_search",
    srcs = ["search.cpp"],
    hdrs = ["search.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_board",
        ":chaturaji_mcts_node",
        ":chaturaji_model",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Depends on the selector library
    ],
)

cc_library(
    name = "chaturaji_self_play",
    srcs = ["self_play.cpp"],
    hdrs = ["self_play.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_board",
        ":chaturaji_mcts_node",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Depends on the selector library
    ],
)

cc_library(
    name = "chaturaji_training",
    srcs = ["train.cpp"],
    hdrs = ["train.h"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_model",
        ":chaturaji_self_play",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Depends on the selector library
    ],
)

# === Main Executable ===

cc_binary(
    name = "chaturaji_engine",
    srcs = ["main.cpp"],
    copts = MSVC_CXX17_COPTS,
    deps = [
        ":chaturaji_board",
        ":chaturaji_model",
        ":chaturaji_search",
        ":chaturaji_training",
        ":chaturaji_types",
        ":chaturaji_utils",
        ":libtorch_configured", # Depends on the selector library
    ],
)

# === Testing ===

cc_test(
    name = "zobrist_test",
    srcs = ["zobrist_test.cpp"],
    copts = MSVC_CXX17_COPTS, # Use your common compiler options
    deps = [
        ":chaturaji_board", # Depends on the board library
        ":chaturaji_types", # Depends on types
        # Add other dependencies if needed (e.g., utils if used)
    ],
    # Link statically if preferred, or rely on runtime DLLs
    linkstatic = True,
)