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

# Import core libraries (interface and shared)
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

# Import additional required shared libraries (DLLs)
# These are needed at runtime and will be copied by Bazel
cc_import(
    name = "uv_dll",
    shared_library = "lib/uv.dll",
)
cc_import(
    name = "torch_global_deps_dll",
    shared_library = "lib/torch_global_deps.dll",
)
cc_import(
    name = "fbgemm_dll",
    shared_library = "lib/fbgemm.dll",
)
cc_import(
    name = "asmjit_dll",
    shared_library = "lib/asmjit.dll",
)
cc_import(
    name = "libiompstubs5md_dll",
    shared_library = "lib/libiompstubs5md.dll",
)
cc_import(
    name = "libiomp5md_dll",
    shared_library = "lib/libiomp5md.dll",
)


cc_library(
    name = "libtorch",
    hdrs = glob(["include/**"]), # Simplified glob
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    # Add ALL imported libraries (static/interface and shared) as dependencies.
    # This ensures Bazel copies the shared libraries to the runfiles.
    deps = [
        ":torch_lib",
        ":torch_cpu_lib",
        ":c10_lib",
        ":uv_dll",               
        ":torch_global_deps_dll",
        ":fbgemm_dll",           
        ":asmjit_dll",           
        ":libiompstubs5md_dll",  
        ":libiomp5md_dll",       
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

# Import core libraries (interface and shared).
cc_import(
    name = "torch_lib_debug",
    interface_library = "lib/torch.lib",     
    shared_library = "lib/torch.dll",        
)
cc_import(
    name = "torch_cpu_lib_debug",
    interface_library = "lib/torch_cpu.lib", 
    shared_library = "lib/torch_cpu.dll",    
)
cc_import(
    name = "c10_lib_debug",
    interface_library = "lib/c10.lib",       
    shared_library = "lib/c10.dll",          
)

# Import additional required shared libraries (DLLs) for debug.
cc_import(
    name = "uv_dll_debug",
    shared_library = "lib/uv.dll", 
)
cc_import(
    name = "torch_global_deps_dll_debug",
    shared_library = "lib/torch_global_deps.dll", 
)
cc_import(
    name = "fbgemm_dll_debug",
    shared_library = "lib/fbgemm.dll", 
)
cc_import(
    name = "asmjit_dll_debug",
    shared_library = "lib/asmjit.dll", 
)
cc_import(
    name = "libiompstubs5md_dll_debug",
    shared_library = "lib/libiompstubs5md.dll", 
)
cc_import(
    name = "libiomp5md_dll_debug",
    shared_library = "lib/libiomp5md.dll", # Corrected name
)

cc_library(
    name = "libtorch_debug",
    hdrs = glob(["include/**"]), # Simplified glob
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    # Add ALL imported libraries (static/interface and shared) as dependencies.
    # This ensures Bazel copies the shared libraries to the runfiles.
    deps = [
        ":torch_lib_debug",
        ":torch_cpu_lib_debug",
        ":c10_lib_debug",
        ":uv_dll_debug",               
        ":torch_global_deps_dll_debug",
        ":fbgemm_dll_debug",           
        ":asmjit_dll_debug",           
        ":libiompstubs5md_dll_debug",  
        ":libiomp5md_dll_debug",       
    ],
)
""",
)

# --- End Libtorch Configuration ---