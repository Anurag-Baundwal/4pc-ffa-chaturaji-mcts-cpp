import torch
import onnxruntime as ort
import sys

def check_pytorch_xpu():
    print(f"\n--- Checking PyTorch (Training) ---")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 1. Check for XPU (Native Intel Support)
    try:
        # Note: In very new versions, this might be standard torch.xpu
        # In slightly older 'native' builds, it might still need an import or check
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_name = torch.xpu.get_device_name(0)
            print(f"SUCCESS: Intel XPU detected: {device_name}")
            
            # 2. Test Tensor Creation
            t = torch.ones(2, 2).to("xpu")
            print(f"Tensor on XPU: {t.device}")
            
            # 3. Test FP16 Support
            try:
                t_half = t.half()
                print("SUCCESS: FP16 (Half) precision supported on XPU.")
            except Exception as e:
                print(f"WARNING: FP16 not supported: {e}")
            
            # # Instead of t.half()
            # try:
            #     t_bf16 = t.to(torch.bfloat16)
            #     print("SUCCESS: BFloat16 supported.")
            # except Exception as e:
            #     print(f"BFloat16 failed: {e}")
                
        else:
            print("FAILURE: Native XPU not detected.")
            print("Did you install from the --index-url https://download.pytorch.org/whl/xpu ?")
            
    except Exception as e:
        print(f"ERROR checking XPU: {e}")

    # Fallback check for DirectML (If you decided to install it as backup)
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"DirectML Device available: {dml_device}")
    except ImportError:
        print("DirectML not installed (this is optional fallback).")

def check_onnx_openvino():
    print(f"\n--- Checking ONNX Runtime (Inference) ---")
    print(f"ORT Version: {ort.__version__}")
    
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    
    if 'OpenVINOExecutionProvider' in providers:
        print("SUCCESS: OpenVINO Execution Provider is available.")
    else:
        print("WARNING: OpenVINO provider NOT found.")
        print("Note: This only affects Python. C++ requires specific DLL linking.")

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    check_pytorch_xpu()
    check_onnx_openvino()