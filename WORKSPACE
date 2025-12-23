# WORKSPACE (or WORKSPACE.bazel)

workspace(name = "chaturaji_cpp_project")

new_local_repository(
    name = "onnxruntime",
    path = "C:/onnxruntime-win-x64-1.23.2", 
    build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
package(default_visibility = ["//visibility:public"])

cc_import(
    name = "onnxruntime_lib",
    interface_library = "lib/onnxruntime.lib",
    shared_library = "lib/onnxruntime.dll",
)

cc_library(
    name = "onnxruntime",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    deps = [":onnxruntime_lib"],
)
""",
)