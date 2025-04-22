# WORKSPACE (or WORKSPACE.bazel)

workspace(name = "chaturaji_cpp_project")

# --- Libtorch Configuration for Windows CPU ---

# --- RELEASE VERSION ---
new_local_repository(
    name = "libtorch_release",
    path = "C:/Users/dell3/Downloads/libtorch-win-shared-with-deps-2.6.0+cpu/libtorch",
    build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "torch_lib",
    interface_library = "lib/torch.lib",
    shared_library = "lib/torch.dll",
)
cc_import(
    name = "torch_cpu_lib",
    interface_library = "lib/torch_cpu.lib",
    shared_library = "lib/torch_cpu.dll",
)
cc_import(
    name = "c10_lib",
    interface_library = "lib/c10.lib",
    shared_library = "lib/c10.dll",
)
# Add other cc_import statements for deps if needed

cc_library(
    name = "libtorch",
    hdrs = glob(["include/**"]), # Simplified glob
    # Provide BOTH the base 'include' and the specific API include path
    includes = [
        "include",                          # For resolving nested includes like torch/csrc/...
        "include/torch/csrc/api/include",   # For resolving top-level includes like torch/torch.h
    ], # <--- CORRECTED PATHS (List of paths)
    deps = [
        ":torch_lib",
        ":torch_cpu_lib",
        ":c10_lib",
        # Add other imported deps here if needed
    ],
)
""",
)

# --- DEBUG VERSION ---
new_local_repository(
    name = "libtorch_debug",
    path = "C:/Users/dell3/Downloads/libtorch-win-shared-with-deps-debug-2.6.0+cpu/libtorch",
    build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "torch_lib_debug",
    interface_library = "lib/torch_debug.lib",
    shared_library = "lib/torch_debug.dll",
)
cc_import(
    name = "torch_cpu_lib_debug",
    interface_library = "lib/torch_cpu_debug.lib",
    shared_library = "lib/torch_cpu_debug.dll",
)
cc_import(
    name = "c10_lib_debug",
    interface_library = "lib/c10_debug.lib",
    shared_library = "lib/c10_debug.dll",
)
# Add other cc_import statements for deps if needed

cc_library(
    name = "libtorch_debug",
    hdrs = glob(["include/**"]), # Simplified glob
    # Provide BOTH the base 'include' and the specific API include path
    includes = [
        "include",                          # For resolving nested includes like torch/csrc/...
        "include/torch/csrc/api/include",   # For resolving top-level includes like torch/torch.h
    ], # <--- CORRECTED PATHS (List of paths)
    deps = [
        ":torch_lib_debug",
        ":torch_cpu_lib_debug",
        ":c10_lib_debug",
        # Add other imported debug deps here if needed
    ],
)
""",
)

# --- End Libtorch Configuration ---