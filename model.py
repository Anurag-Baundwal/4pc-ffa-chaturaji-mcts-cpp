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

# --- Input/Output Configuration ---
# Spatial Input: 28 Planes
# 20 (Pieces: 5 types * 4 players) + 4 (X-Ray Attacks) + 4 (Standard Attacks)
NUM_INPUT_PLANES = 28 

# Scalar Input: 34 Scalars
# 4(Material) + 4(PawnCnt) + 4(Conn) + 4(Dist) + 4(NumSafeSq) + 
# 4(Active) + 4(Points) + 1(50mv) + 4(Check) + 1(OppCnt)
NUM_INPUT_SCALARS = 34

POLICY_OUTPUT_SIZE = 4096 
VALUE_OUTPUT_SIZE = 4     

# Network Architecture
NUM_RES_BLOCKS = 8
NUM_CHANNELS = 64

# Policy Head Configuration
POLICY_HEAD_CONV_CHANNELS = 32 
# We use an intermediate mixing layer to allow scalars to modulate spatial features
POLICY_MIXING_DIM = 2048 

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
    - Input 1 (Spatial): [Batch, 28, 8, 8] -> Processed by ResNet Backbone
    - Input 2 (Scalars): [Batch, 34]       -> Injected into Heads
    
    - Trunk: Conv + BatchNorm + SiLU -> 8 Residual Blocks (processes spatial only)
    
    - Policy Head: 
      1. Spatial Features -> Conv(32) -> BN -> SiLU -> Flatten (Size 2048)
      2. Concatenate Scalars (Size 2048 + 34 = 2082)
      3. Mixing Layer: Linear(2082 -> 2048) -> SiLU
      4. Linear -> Logits(4096)
      
    - Value Head: 
      1. Spatial Features -> Conv(24) -> BN -> SiLU -> Flatten (Size 1536)
      2. Concatenate Scalars (Size 1536 + 34 = 1570)
      3. Linear(1570 -> 384) -> SiLU -> Linear(256) -> SiLU -> Linear(4) -> Tanh
    """
    def __init__(self):
        super().__init__()
        # --- Backbone (Trunk) ---
        # Takes spatial planes only
        self.conv1 = nn.Conv2d(NUM_INPUT_PLANES, NUM_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)

        self.resblocks = nn.ModuleList([
            ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CONV_CHANNELS)
        
        # Calculate sizes
        self.policy_spatial_flat_size = POLICY_HEAD_CONV_CHANNELS * BOARD_AREA # 32 * 64 = 2048
        self.policy_combined_size = self.policy_spatial_flat_size + NUM_INPUT_SCALARS
        
        # Mixing Layer: Allows scalars to non-linearly interact with spatial features
        self.policy_mixing = nn.Linear(self.policy_combined_size, POLICY_MIXING_DIM)
        # Final projection
        self.policy_fc = nn.Linear(POLICY_MIXING_DIM, POLICY_OUTPUT_SIZE)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CONV_CHANNELS)
        
        # Calculate sizes
        self.value_spatial_flat_size = VALUE_HEAD_CONV_CHANNELS * BOARD_AREA # 24 * 64 = 1536
        self.value_combined_size = self.value_spatial_flat_size + NUM_INPUT_SCALARS
        
        # Value head already has depth, so we inject into the first FC layer
        self.value_fc1 = nn.Linear(self.value_combined_size, VALUE_FC_HIDDEN_CHANNELS)
        self.value_fc_mid = nn.Linear(VALUE_FC_HIDDEN_CHANNELS, VALUE_FC_MID_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_FC_MID_CHANNELS, VALUE_OUTPUT_SIZE)

        # --- Uncertainty Weighting Parameters (Poor Man's GradNorm) ---
        # These learnable scalars balance the Policy and Value losses dynamically.
        # log_vars[0] -> Policy, log_vars[1] -> Value.
        # Initial s_p = 0 (weight 1.0), Initial s_v = -2.0 (weight ~3.7)
        self.log_vars = nn.Parameter(torch.tensor([0.0, -2.0]))

    def forward(self, x_planes, x_scalars):
        # --- Backbone (Spatial) ---
        x = F.silu(self.bn1(self.conv1(x_planes)))
        for block in self.resblocks:
            x = block(x)

        # --- Policy Head Forward ---
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.silu(p)
        p = p.flatten(1) 
        
        # Concatenate Global Scalars
        p = torch.cat([p, x_scalars], dim=1)
        
        # Non-linear mixing
        p = F.silu(self.policy_mixing(p))
        
        # Final Logits
        p = self.policy_fc(p)

        # --- Value Head Forward ---
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.silu(v)
        v = v.flatten(1)
        
        # Concatenate Global Scalars
        v = torch.cat([v, x_scalars], dim=1)
        
        # Value MLP
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
        
        # Patch Inputs
        # input[0] -> input_planes
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        # input[1] -> input_scalars
        model.graph.input[1].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
        
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
    dummy_planes = torch.randn(1, NUM_INPUT_PLANES, BOARD_DIM, BOARD_DIM).to(device)
    dummy_scalars = torch.randn(1, NUM_INPUT_SCALARS).to(device)

    # 1. Prepare Arguments
    export_args = {
        "model": model,
        "args": (dummy_planes, dummy_scalars),
        "f": output_path,
        "export_params": True,
        "opset_version": 18,
        "do_constant_folding": False, 
        "input_names": ['input_planes', 'input_scalars'],
        "output_names": ['policy', 'value'],
        "dynamic_axes": {
            'input_planes': {0: 'batch_size'},
            'input_scalars': {0: 'batch_size'},
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