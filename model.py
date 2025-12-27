import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Board Dimensions & NN Configuration ---
BOARD_DIM = 8
BOARD_AREA = 64

# Input/Output Dimensions
NUM_INPUT_CHANNELS = 37 # Pieces + Meta + Attacks
POLICY_OUTPUT_SIZE = 4096 # 64 * 64
VALUE_OUTPUT_SIZE = 4     # 4 Players

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
        # Input: NUM_INPUT_CHANNELS (34)
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNELS, NUM_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)

        self.resblocks = nn.ModuleList([
            ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CONV_CHANNELS)
        # 8*8 board size * policy channels
        self.policy_fc = nn.Linear(POLICY_HEAD_CONV_CHANNELS * BOARD_AREA, POLICY_OUTPUT_SIZE)

        # Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CONV_CHANNELS * BOARD_AREA, VALUE_FC_HIDDEN_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_FC_HIDDEN_CHANNELS, VALUE_OUTPUT_SIZE)

    def forward(self, x):
        # Body
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)

        # Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)      # Logits (no softmax here, CrossEntropyLoss handles it)

        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        v = torch.tanh(v)          # Output range [-1, 1]

        return p, v

def export_for_cpp(model_path, output_path):
    """
    Loads a PyTorch model and traces it so C++ Libtorch can load it.
    This bridge allows you to train in Python but run inference in your current C++ engine.
    """
    model = ChaturajiNN()
    # Load weights if they exist, else use random init
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        print("Model file not found, exporting random initialization.")
        model.eval()

    # Create dummy input: (Batch=1, Channels=NUM_INPUT_CHANNELS, H=8, W=8)
    dummy_input = torch.rand(1, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM)
    
    # Trace the model
    traced_script_module = torch.jit.trace(model, dummy_input)
    
    # Save
    traced_script_module.save(output_path)
    print(f"Exported JIT model to {output_path}")

def export_to_onnx(model_path, output_path):
    """
    Loads weights from model_path and exports to ONNX format at output_path.
    """
    print(f"Exporting ONNX: Loading weights from {model_path}...")
    
    # --- Device Selection ---
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # -------------------------------------------

    model = ChaturajiNN().to(device) # Move model to device
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("Warning: Model weights not found. Exporting with random initialization.")
    
    model.eval()

    # Dummy input: [Batch=1, Channels=NUM_INPUT_CHANNELS, Height=8, Width=8]
    dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=18,          # 
        do_constant_folding=True,  # Optimization
        input_names=['input'],     # The name the C++ code will use to feed data
        output_names=['policy', 'value'], # The names C++ will use to fetch results
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow variable batch size
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported ONNX model to {output_path}")

if __name__ == "__main__":
    import sys
    
    # Usage: 
    # 1. python model.py export <input.pth> <output.onnx>
    # 2. python model.py export_random <output.onnx>
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "export":
            input_pth = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
            output_onnx = sys.argv[3] if len(sys.argv) > 3 else "model.onnx"
            export_to_onnx(input_pth, output_onnx)
            
        elif cmd == "export_random":
            output_onnx = sys.argv[2] if len(sys.argv) > 2 else "initial_random.onnx"
            print(f"Exporting random initialized model to {output_onnx}...")
            # Initialize random model
            model = ChaturajiNN()
            model.eval()
            dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM)
            torch.onnx.export(
                model, dummy_input, output_onnx,
                export_params=True, opset_version=18, do_constant_folding=True,
                input_names=['input'], output_names=['policy', 'value'],
                dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
            )
            print("Done.")
    else:
        # Standard smoke test
        net = ChaturajiNN()
        dummy = torch.randn(2, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM)
        p, v = net(dummy)
        print("--- Smoke Test ---")
        print(f"Policy Shape: {p.shape}") # Should be [2, 4096]
        print(f"Value Shape:  {v.shape}") # Should be [2, 4]