import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx 
import warnings

try:
    from onnxconverter_common import float16
    HAS_FP16_CONVERTER = True
except ImportError:
    HAS_FP16_CONVERTER = False

# --- Board Dimensions & NN Configuration ---
BOARD_DIM = 8
BOARD_AREA = 64

# Input/Output Dimensions
NUM_INPUT_CHANNELS = 37 
POLICY_OUTPUT_SIZE = 4096 
VALUE_OUTPUT_SIZE = 4     

# Network Architecture
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 64

# Policy Head Configuration
POLICY_HEAD_CONV_CHANNELS = 32 

# Value Head Configuration
VALUE_HEAD_CONV_CHANNELS = 24
VALUE_FC_HIDDEN_CHANNELS = 384
VALUE_FC_MID_CHANNELS = 256
NUM_VALUE_OUTPUTS = 4

class ResBlock(nn.Module):
    """
    Standard Residual Block using SiLU (Swish) activation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.silu(out)
        return out

class ChaturajiNN(nn.Module):
    """
    AlphaZero-style network adapted for 4-player Chaturaji.
    
    Architecture:
    - Input: [Batch, 37, 8, 8]
    - Trunk: Conv + BatchNorm + SiLU -> 10 Residual Blocks
    - Policy Head: Conv(32) -> BN -> SiLU -> Linear -> Logits(4096)
    - Value Head: Conv(24) -> BN -> SiLU -> Linear -> Linear -> Linear -> Tanh(4)
    """
    def __init__(self):
        super().__init__()
        # --- Backbone (Trunk) ---
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNELS, NUM_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)

        self.resblocks = nn.ModuleList([
            ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CONV_CHANNELS)
        # Input features: 32 channels * 64 squares = 2048
        self.policy_fc = nn.Linear(POLICY_HEAD_CONV_CHANNELS * BOARD_AREA, POLICY_OUTPUT_SIZE)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CONV_CHANNELS * BOARD_AREA, VALUE_FC_HIDDEN_CHANNELS)
        self.value_fc_mid = nn.Linear(VALUE_FC_HIDDEN_CHANNELS, VALUE_FC_MID_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_FC_MID_CHANNELS, VALUE_OUTPUT_SIZE)

    def forward(self, x):
        # Backbone
        x = F.silu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)

        # --- Policy Head Forward ---
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.silu(p)
        p = p.flatten(1) 
        p = self.policy_fc(p)

        # --- Value Head Forward ---
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.silu(v)
        v = v.flatten(1)
        v = F.silu(self.value_fc1(v))
        v = F.silu(self.value_fc_mid(v))
        v = self.value_fc2(v)
        v = torch.tanh(v)

        return p, v

def force_patch_onnx_batch_size(model_path):
    """
    Manually overrides input/output dimensions to 'batch_size'.
    """
    try:
        model = onnx.load(model_path)
        
        # Patch Input
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        # Patch Outputs
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        model.graph.output[1].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        
        onnx.save(model, model_path)
        print(f"[Python] Manually patched ONNX dimensions for {model_path}")
    except Exception as e:
        print(f"[Python] Warning: Failed to patch ONNX dimensions: {e}")

def export_to_onnx(model_path, output_path, random_init=False):
    """
    Exports the PyTorch model to ONNX format.
    If random_init is True, exports a model with random weights instead of loading from disk.
    Also attempts to create an FP16 copy for NVIDIA GPU inference.
    """
    if random_init:
        print(f"Exporting random initialized model to {output_path}...")
    else:
        print(f"Exporting ONNX: Loading weights from {model_path}...")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ChaturajiNN().to(device)
    
    if not random_init:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            print("Warning: Model weights not found. Exporting random init.")
    
    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM).to(device)

    # 1. Prepare Arguments
    export_args = {
        "model": model,
        "args": dummy_input,
        "f": output_path,
        "export_params": True,
        "opset_version": 18,
        "do_constant_folding": False, 
        "input_names": ['input'],
        "output_names": ['policy', 'value'],
        "dynamic_axes": {
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    }

    # 2. Attempt export (handling legacy vs dynamo exporter versions)
    try:
        torch.onnx.export(**export_args, dynamo=False)
        print("[Python] Exported using Legacy Exporter (dynamo=False).")
    except TypeError:
        print("[Python] 'dynamo' arg not supported, calling standard export.")
        torch.onnx.export(**export_args)

    # 3. Apply Patch to the main FP32 model
    force_patch_onnx_batch_size(output_path)
    print(f"Successfully exported FP32 ONNX model to {output_path}")

    # 4. Attempt FP16 Conversion
    if HAS_FP16_CONVERTER:
        try:
            print("[Python] Attempting to create FP16 model...")
            
            # suppress truncation warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="the float32 number.*will be truncated")
                
                model_fp32 = onnx.load(output_path)
                model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
                
                fp16_path = output_path.replace(".onnx", "_fp16.onnx")
                onnx.save(model_fp16, fp16_path)
                print(f"[Python] Successfully exported FP16 ONNX model to {fp16_path}")

        except Exception as e:
            print(f"[Python] Warning: FP16 conversion failed: {e}")
    else:
        print("[Python] 'onnxconverter-common' not installed. Skipping FP16 export.")
        print("         (To enable: pip install onnxconverter-common)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "export":
            input_pth = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
            output_onnx = sys.argv[3] if len(sys.argv) > 3 else "model.onnx"
            export_to_onnx(input_pth, output_onnx, random_init=False)
            
        elif cmd == "export_random":
            output_onnx = sys.argv[2] if len(sys.argv) > 2 else "initial_random.onnx"
            export_to_onnx(None, output_onnx, random_init=True)
            print("Done.")