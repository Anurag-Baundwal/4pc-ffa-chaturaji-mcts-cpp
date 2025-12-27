# WORKSPACE (or WORKSPACE.bazel)

workspace(name = "chaturaji_cpp_project")

new_local_repository(
    name = "onnxruntime",
    path = "C:/onnxruntime-openvino",  # <--- Point to the folder you extracted
    build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

# 1. The Import Library (.lib) used at Link Time
cc_import(
    name = "onnxruntime_lib",
    # Check your folder! It is often in 'runtimes/win-x64/native' for NuGet packages
    interface_library = "runtimes/win-x64/native/onnxruntime.lib", 
    
    # The Runtime Library (.dll) used at Run Time
    shared_library = "runtimes/win-x64/native/onnxruntime.dll",
)

# 2. The Main Library Target
cc_library(
    name = "onnxruntime",
    # Headers are usually deep in the 'build' folder for NuGet
    hdrs = glob(["build/native/include/**"]),
    
    # Tell compiler to add this folder to include path
    includes = ["build/native/include"], 
    
    deps = [":onnxruntime_lib"],
)
""",
)