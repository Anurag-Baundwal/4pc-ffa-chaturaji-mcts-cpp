import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx 

# --- Board Dimensions & NN Configuration ---
BOARD_DIM = 8
BOARD_AREA = 64

# Input/Output Dimensions
NUM_INPUT_CHANNELS = 37 
POLICY_OUTPUT_SIZE = 4096 
VALUE_OUTPUT_SIZE = 4     

# Network Architecture
NUM_RES_BLOCKS = 4
NUM_CHANNELS = 64
POLICY_HEAD_CONV_CHANNELS = 8
VALUE_HEAD_CONV_CHANNELS = 12
VALUE_FC_HIDDEN_CHANNELS = 256
NUM_VALUE_OUTPUTS = 4

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChaturajiNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNELS, NUM_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)

        self.resblocks = nn.ModuleList([
            ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CONV_CHANNELS)
        self.policy_fc = nn.Linear(POLICY_HEAD_CONV_CHANNELS * BOARD_AREA, POLICY_OUTPUT_SIZE)

        # Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CONV_CHANNELS * BOARD_AREA, VALUE_FC_HIDDEN_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_FC_HIDDEN_CHANNELS, VALUE_OUTPUT_SIZE)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)

        # Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        # FIX: Use flatten(1) instead of view(size(0), -1) to avoid tracing static batch size
        p = p.flatten(1) 
        p = self.policy_fc(p)

        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        # FIX: Use flatten(1)
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
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

def export_to_onnx(model_path, output_path):
    print(f"Exporting ONNX: Loading weights from {model_path}...")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ChaturajiNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("Warning: Model weights not found. Exporting random init.")
    model.eval()

    # Use a dummy batch size of 1. 
    # Because we use flatten(1) and dynamic_axes, this should not stick.
    dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM).to(device)

    # 1. Prepare Arguments
    export_args = {
        "model": model,
        "args": dummy_input,
        "f": output_path,
        "export_params": True,
        "opset_version": 18,        # Use 18 as requested by logs
        "do_constant_folding": False, # Ensure dynamic calc of shapes if needed
        "input_names": ['input'],
        "output_names": ['policy', 'value'],
        "dynamic_axes": {
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    }

    # 2. Attempt to force legacy exporter via kwarg if supported
    # Recent PyTorch versions introduced 'dynamo' arg to toggle the new exporter
    try:
        torch.onnx.export(**export_args, dynamo=False)
        print("[Python] Exported using Legacy Exporter (dynamo=False).")
    except TypeError:
        # Fallback for versions where 'dynamo' arg doesn't exist
        print("[Python] 'dynamo' arg not supported, calling standard export.")
        torch.onnx.export(**export_args)

    # 3. Apply Patch
    force_patch_onnx_batch_size(output_path)
    print(f"Successfully exported ONNX model to {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "export":
            input_pth = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
            output_onnx = sys.argv[3] if len(sys.argv) > 3 else "model.onnx"
            export_to_onnx(input_pth, output_onnx)
            
        elif cmd == "export_random":
            output_onnx = sys.argv[2] if len(sys.argv) > 2 else "initial_random.onnx"
            print(f"Exporting random initialized model to {output_onnx}...")
            
            # Re-use logic for random export
            model = ChaturajiNN()
            model.eval()
            dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM)
            
            export_args = {
                "model": model,
                "args": dummy_input,
                "f": output_onnx,
                "export_params": True,
                "opset_version": 18,
                "do_constant_folding": False,
                "input_names": ['input'],
                "output_names": ['policy', 'value'],
                "dynamic_axes": {'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
            }
            
            try:
                torch.onnx.export(**export_args, dynamo=False)
            except TypeError:
                torch.onnx.export(**export_args)
                
            force_patch_onnx_batch_size(output_onnx)
            print("Done.")