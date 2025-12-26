import torch
import torch.optim as optim
import torch.nn.functional as F
import struct
import os
import glob
import argparse
import sys
import numpy as np

from model import (
    ChaturajiNN, export_to_onnx,
    NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_AREA,
    POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
)

# Constants for binary data parsing
INPUT_SIZE = NUM_INPUT_CHANNELS * BOARD_AREA
POLICY_SIZE = POLICY_OUTPUT_SIZE
VALUE_SIZE = VALUE_OUTPUT_SIZE
SAMPLE_SIZE_BYTES = (INPUT_SIZE + POLICY_SIZE + VALUE_SIZE) * 4

class ReplayBuffer:
    def __init__(self, data_dir, max_size):
        self.data_dir = data_dir
        self.max_size = max_size
        self.states = torch.empty(0)
        self.policies = torch.empty(0)
        self.values = torch.empty(0)
        self.load_buffer()

    def load_buffer(self):
        files = glob.glob(os.path.join(self.data_dir, "gen_*.bin"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        t_s, t_p, t_v = [], [], []
        total = 0
        
        # Calculate floats per sample once
        floats_per_sample = INPUT_SIZE + POLICY_SIZE + VALUE_SIZE
        
        for fp in files:
            if total >= self.max_size: break
            
            # Get file size to calculate n samples without reading content yet
            file_size = os.path.getsize(fp)
            n = file_size // SAMPLE_SIZE_BYTES
            
            if n == 0: continue
            
            with open(fp, "rb") as f:
                content = f.read()
            
            # OPTIMIZATION: Use frombuffer instead of struct.unpack
            # This reads directly from C memory, avoiding massive Python tuple creation
            data = np.frombuffer(content, dtype=np.float32).reshape(n, -1)
            
            # Create tensors (using copy to ensure memory is contiguous and separate from the raw buffer)
            # We use .clone().detach() or torch.tensor() to ensure we own the memory
            t_s.append(torch.from_numpy(data[:, :INPUT_SIZE].copy()).view(n, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM))
            t_p.append(torch.from_numpy(data[:, INPUT_SIZE : INPUT_SIZE + POLICY_SIZE].copy()))
            t_v.append(torch.from_numpy(data[:, INPUT_SIZE + POLICY_SIZE :].copy()))
            
            total += n
            
            # Explicitly delete raw content to free RAM immediately for the next iteration
            del content
            del data

        if total > 0:
            # Slice strictly to max_size to avoid over-allocating VRAM later
            self.states = torch.cat(t_s)[:self.max_size]
            self.policies = torch.cat(t_p)[:self.max_size]
            self.values = torch.cat(t_v)[:self.max_size]
            print(f"[Python] Replay Buffer: {self.states.size(0)} samples loaded.")

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.states.size(0), (batch_size,))
        return self.states[idx], self.policies[idx], self.values[idx]

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define save paths directly in the target directory
    model_pth = os.path.join(args.save_dir, "latest.pth")
    opt_pth = os.path.join(args.save_dir, "latest.optimizer.pth")
    onnx_path = os.path.join(args.save_dir, "latest.onnx")

    # 1. Model & Optimizer Setup
    model = ChaturajiNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 2. Loading Logic
    if args.load_weights:
        # Determine .pth path (if user pointed to .onnx, look for .pth)
        target_pth = args.load_weights.replace(".onnx", ".pth") if args.load_weights.endswith(".onnx") else args.load_weights
        target_opt = target_pth.replace(".pth", ".optimizer.pth")

        if not os.path.exists(target_pth):
            print(f"[Python] FATAL ERROR: Weights file not found: {target_pth}")
            print(f"[Python] Training cannot resume without the .pth weights file.")
            print(f"[Python] To start fresh with random weights, remove the --load-model argument.")
            sys.exit(1)

        # Load Weights
        model.load_state_dict(torch.load(target_pth, map_location=device))
        print(f"[Python] LOADED: Weights from {target_pth}")

        # Load Optimizer
        if os.path.exists(target_opt):
            optimizer.load_state_dict(torch.load(target_opt, map_location=device))
            print(f"[Python] LOADED: Optimizer state from {target_opt}")
        else:
            print(f"[Python] NOTICE: Optimizer file {target_opt} not found. Resetting Adam optimizer.")
    else:
        print("[Python] NOTICE: No load-model specified. Starting from SCRATCH (Random Init).")

    # Data check
    buffer = ReplayBuffer(args.data_dir, args.max_buffer_size)
    
    if buffer.states.size(0) < args.batch_size or args.new_samples == 0:
        print("[Python] Not enough data. Skipping training.")
        return

    num_steps = int((args.new_samples * args.sampling_rate) / args.batch_size)
    if num_steps == 0:
        print("[Python] New samples too low for a full batch. Skipping training.")
        return

    print(f"[Python] New Samples: {args.new_samples}, Sampling Rate: {args.sampling_rate}")
    print(f"[Python] Training for {num_steps} steps...")

    # Training
    model.train()
    for step in range(num_steps):
        s, tp, tv = buffer.sample_batch(args.batch_size)
        s, tp, tv = s.to(device), tp.to(device), tv.to(device)
        
        optimizer.zero_grad()
        p, v = model(s)
        
        # Explicit loss components
        loss_policy = -torch.sum(tp * F.log_softmax(p, dim=1), dim=1).mean()
        loss_value = F.mse_loss(v, tv)
        loss = loss_policy + loss_value
        
        loss.backward()
        optimizer.step()
        
        # Individual component reporting for every step
        print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.4f} "
              f"(Policy: {loss_policy.item():.4f}, Value: {loss_value.item():.4f})")

    # Save Checkpoint
    torch.save(model.state_dict(), model_pth)
    torch.save(optimizer.state_dict(), opt_pth)
    export_to_onnx(model_pth, onnx_path)
    print(f"[Python] Training complete. Saved checkpoint to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--max-buffer-size", type=int, default=200000)
    parser.add_argument("--new-samples", type=int, default=0)
    parser.add_argument("--sampling-rate", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--load-weights", type=str, default="")
    train_loop(args = parser.parse_args())