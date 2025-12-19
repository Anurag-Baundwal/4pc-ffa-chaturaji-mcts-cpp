import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration matches model.cpp constants
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
        # Input: 33 channels (pieces + aux planes)
        self.conv1 = nn.Conv2d(33, NUM_CHANNELS, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(NUM_CHANNELS)

        self.resblocks = nn.ModuleList([
            ResBlock(NUM_CHANNELS) for _ in range(NUM_RES_BLOCKS)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(POLICY_HEAD_CONV_CHANNELS)
        # 8*8 board size * policy channels
        self.policy_fc = nn.Linear(POLICY_HEAD_CONV_CHANNELS * 8 * 8, 4096)

        # Value Head
        self.value_conv = nn.Conv2d(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, kernel_size=1, bias=True)
        self.value_bn = nn.BatchNorm2d(VALUE_HEAD_CONV_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_HEAD_CONV_CHANNELS * 8 * 8, VALUE_FC_HIDDEN_CHANNELS)
        self.value_fc2 = nn.Linear(VALUE_FC_HIDDEN_CHANNELS, NUM_VALUE_OUTPUTS)

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

    # Create dummy input for tracing (Batch=1, Channels=33, H=8, W=8)
    example_input = torch.rand(1, 33, 8, 8)
    
    # Trace the model
    traced_script_module = torch.jit.trace(model, example_input)
    
    # Save
    traced_script_module.save(output_path)
    print(f"Exported JIT model to {output_path}")

if __name__ == "__main__":
    # Smoke test
    net = ChaturajiNN()
    dummy = torch.randn(2, 33, 8, 8)
    p, v = net(dummy)
    print(f"Policy Shape: {p.shape} (Expected [2, 4096])")
    print(f"Value Shape:  {v.shape} (Expected [2, 4])")