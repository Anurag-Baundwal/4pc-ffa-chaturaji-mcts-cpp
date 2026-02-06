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

# Policy Output: 64 Spatial Planes
# 56 Queen-like moves (8 directions * 7 distances) + 8 Knight moves
NUM_POLICY_PLANES = 64
POLICY_OUTPUT_SIZE = NUM_POLICY_PLANES * BOARD_AREA + 1 # Index 4096 is for resignation
VALUE_OUTPUT_SIZE = 4     

# Network Architecture
NUM_RES_BLOCKS = 6      # How many residual blocks in the trunk
NUM_CHANNELS = 64       # Width of the main trunk (Backbone)
SE_REDUCTION = 2        # Squeeze ratio for SE blocks

# Policy Head Configuration
POLICY_HEAD_CONV_CHANNELS = 24 # Depth of the hidden layer in policy head

# Value Head Configuration
# Optimized for 6x64 architecture to maintain >80% trunk parameter ratio.
# Lc0 Standard: 1x1 Conv (No Stride) -> Flatten -> Dense -> Dense.
VALUE_HEAD_CONV_CHANNELS = 12  # Reduced to 12 to keep the dense layer input manageable
VALUE_FC_HIDDEN_CHANNELS = 96  # Reduced to 96 to prevent head parameter bloat
NUM_VALUE_OUTPUTS = 4

class GlobalSEBlock(nn.Module):
    """
    Global-Context Squeeze-and-Excitation Block.
    Uses global scalars + spatial average pooling to scale convolutional channels.
    """
    def __init__(self, channels, global_channels):
        super().__init__()
        self.input_dim = channels + global_channels
        self.squeeze_dim = channels // SE_REDUCTION
        
        self.fc1 = nn.Linear(self.input_dim, self.squeeze_dim)
        self.fc2 = nn.Linear(self.squeeze_dim, channels)

    def forward(self, x, global_context):
        # x: [Batch, C, H, W], global_context: [Batch, GlobalC]
        batch, c, _, _ = x.size()
        
        # Squeeze: Global Average Pooling [Batch, C]
        w = x.mean(dim=(2, 3))
        
        # Concatenate spatial summary with encoded global scalars
        w = torch.cat([w, global_context], dim=1)
        
        # Excitation: MLP to find channel weights
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        
        # Scale original feature map
        return x * w.view(batch, c, 1, 1)

class ResBlock(nn.Module):
    """
    Standard Residual Block using SiLU (Swish) activation and Global SE.
    """
    def __init__(self, channels, global_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Global-Context Squeeze-and-Excitation
        self.se = GlobalSEBlock(channels, global_channels)

    def forward(self, x, global_context):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply context-aware scaling
        out = self.se(out, global_context)
        
        out += residual
        out = F.silu(out)
        return out

class ChaturajiNN(nn.Module):
    """
    AlphaZero-style network adapted for 4-player Chaturaji.
    
    Architecture:
    - Input 1 (Spatial): [Batch, 28, 8, 8] -> Processed by ResNet Backbone
    - Input 2 (Scalars): [Batch, 34]       -> Injected into SE Blocks and Heads
    
    - Global Encoder: Linear projects 34 scalars to 64 context features (2 layer MLP).
    
    - Trunk: Conv + BatchNorm + SiLU -> 6 Residual Blocks.
      Each block uses the Global context to perform Channel Attention (SE).
    
    - Policy Head (Fully Convolutional / Spatial): 
      1. Spatial Features -> 3x3 Conv(24) -> BN -> SiLU.
         * Maintains spatial awareness across the board.
      2. 1x1 Conv(64 planes).
         * This creates a translational-invariant mapping. If a move pattern is 
           learned in one corner, it is automatically applied to all squares.
         * 64 Planes * 64 Squares = 4096 outputs.
      
    - Value Head (Lc0 Standard 1x1 Conv + Dense): 
      1. Spatial Features -> 1x1 Conv(12) Stride 1 -> BN -> SiLU.
         * Preserves 8x8 spatial resolution (No downsampling).
         * Reduces channels from Trunk(64) to Head(12).
      2. Scalar Gating: Projecting 34 scalars to match 12 channels, gating spatial
         features via sigmoid multiplication.
      3. Flatten (12 * 8 * 8 = 768) -> Concatenate Scalars (Size 802 total).
      4. MLP: 
         - Linear(802 -> 96) -> SiLU
         - Final Output: Linear(96 -> 4) -> Tanh.
    """
    def __init__(self):
        super().__init__()
        
        # --- Global Context Encoder ---
        # Projects scalars into an embedding space used by SE blocks in the backbone
        self.global_encoder = nn.Sequential(
            nn.Linear(NUM_INPUT_SCALARS, NUM_CHANNELS),
            nn.SiLU(),
            nn.Linear(NUM_CHANNELS, NUM_CHANNELS),
            nn.SiLU()
        )

        # --- Backbone (Trunk) ---
        # Takes spatial planes only
        self.conv1 = nn.Conv2d(NUM_INPUT_PLANES, NUM_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)

        self.resblocks = nn.ModuleList([
            ResBlock(NUM_CHANNELS, global_channels=NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # --- Policy Head (Spatial) ---
        # Uses 3x3 Conv to refine features from trunk to prepare for move mapping
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CONV_CHANNELS)
        
        # Project Scalars to match spatial channels
        self.policy_context_projector = nn.Linear(NUM_INPUT_SCALARS, POLICY_HEAD_CONV_CHANNELS)

        # Final Projection: Maps hidden features to 64 "move type" planes.
        self.policy_out = nn.Conv2d(POLICY_HEAD_CONV_CHANNELS, NUM_POLICY_PLANES, kernel_size=1, bias=True)

        # A separate tiny head just for the Resignation logit
        # Takes the 34 global scalars + pooled spatial features -> 1 logit
        self.policy_resign_fc = nn.Linear(POLICY_HEAD_CONV_CHANNELS + NUM_INPUT_SCALARS, 1)

        # --- Value Head (Lc0 Style) ---
        # 1. 1x1 Convolution (No spatial reduction)
        self.value_conv = nn.Conv2d(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CONV_CHANNELS)
        
        # 2. Scalar Gating Projector
        self.value_context_projector = nn.Linear(NUM_INPUT_SCALARS, VALUE_HEAD_CONV_CHANNELS)

        # Sizes: (12 channels * 8 * 8 grid) + raw scalars
        self.value_spatial_flat_size = VALUE_HEAD_CONV_CHANNELS * BOARD_DIM * BOARD_DIM 
        self.value_combined_size = self.value_spatial_flat_size + NUM_INPUT_SCALARS
        
        # 3. Dense Head (Optimized for small model ratio)
        self.value_fc1 = nn.Linear(self.value_combined_size, VALUE_FC_HIDDEN_CHANNELS)
        self.value_fc_out = nn.Linear(VALUE_FC_HIDDEN_CHANNELS, NUM_VALUE_OUTPUTS)

        # --- Uncertainty Weighting Parameters (Poor Man's GradNorm) ---
        # These learnable scalars balance the Policy and Value losses dynamically.
        # log_vars[0] -> Policy, log_vars[1] -> Value.
        self.log_vars = nn.Parameter(torch.tensor([0.0, -2.0]))

    def forward(self, x_planes, x_scalars):
        # --- Global Encoding ---
        global_embed = self.global_encoder(x_scalars)

        # --- Backbone (Spatial) ---
        x = F.silu(self.bn1(self.conv1(x_planes)))
        for block in self.resblocks:
            x = block(x, global_embed)

        # --- Policy Head Forward ---
        # 1. Spatial Processing & Scalar Injection
        p_spatial = F.silu(self.policy_bn(self.policy_conv(x)))
        p_context = self.policy_context_projector(x_scalars).view(-1, POLICY_HEAD_CONV_CHANNELS, 1, 1)
        p_spatial = p_spatial + p_context
        
        # 2. Map to Move Planes (Batch, 64, 8, 8 -> Batch, 4096)
        p_moves = self.policy_out(p_spatial).flatten(1)
        
        # 3. Resignation Logit (Spatial GAP + Scalars -> Batch, 1)
        # Summary of board state mixed with global stats
        p_spatial_gap = p_spatial.mean(dim=(2, 3)) 
        p_resign_input = torch.cat([p_spatial_gap, x_scalars], dim=1)
        p_resign = self.policy_resign_fc(p_resign_input)
        
        # 4. Final Policy (Index 4096 is Resignation)
        p = torch.cat([p_moves, p_resign], dim=1)

        # --- Value Head Forward ---
        # 1. 1x1 Convolution
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.silu(v)
        
        # 2. Scalar Gating (Context-Aware Modulation)
        # Project scalars to [Batch, 12] -> View as [Batch, 12, 1, 1]
        context_gate = torch.sigmoid(self.value_context_projector(x_scalars))
        v = v * context_gate.view(-1, VALUE_HEAD_CONV_CHANNELS, 1, 1)
        
        # Flatten [Batch, 12, 8, 8] -> [Batch, 768]
        v = v.flatten(1)
        
        # 3. Concatenate Global Scalars
        v = torch.cat([v, x_scalars], dim=1)
        
        # 4. Dense MLP
        v = F.silu(self.value_fc1(v))
        v = self.value_fc_out(v)
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

    # Dummy inputs for tracing
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