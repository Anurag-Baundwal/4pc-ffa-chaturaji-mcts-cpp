import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import struct
import os
import glob
import argparse
import random
import numpy as np
from model import ChaturajiNN, export_for_cpp, export_to_onnx 

# Data dimensions
INPUT_SIZE = 33 * 8 * 8
POLICY_SIZE = 4096
VALUE_SIZE = 4
SAMPLE_SIZE_BYTES = (INPUT_SIZE + POLICY_SIZE + VALUE_SIZE) * 4 # 4 bytes per float

class ReplayBuffer:
    def __init__(self, data_dir, max_size):
        self.data_dir = data_dir
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []
        
        self.load_buffer()

    def load_buffer(self):
        # 1. Get all bin files
        files = glob.glob(os.path.join(self.data_dir, "gen_*.bin"))
        
        # 2. Sort by modification time, NEWEST FIRST
        files.sort(key=os.path.getmtime, reverse=True)
        
        total_samples = 0
        loaded_files = 0
        
        print(f"Checking {len(files)} available data files...")
        
        # 3. Load files until we hit max_size
        # We process files from newest to oldest
        temp_states = []
        temp_policies = []
        temp_values = []

        for filepath in files:
            if total_samples >= self.max_size:
                break
                
            with open(filepath, "rb") as f:
                content = f.read()
                
            num_samples_in_file = len(content) // SAMPLE_SIZE_BYTES
            if num_samples_in_file == 0:
                continue

            # Unpack entire file at once into numpy arrays for speed
            # then convert to tensor chunks
            floats = struct.unpack(f'{num_samples_in_file * (INPUT_SIZE + POLICY_SIZE + VALUE_SIZE)}f', content)
            np_data = np.array(floats, dtype=np.float32)
            np_data = np_data.reshape(num_samples_in_file, INPUT_SIZE + POLICY_SIZE + VALUE_SIZE)

            # Split columns
            # State: [N, 2112]
            s = torch.from_numpy(np_data[:, :INPUT_SIZE]).view(num_samples_in_file, 33, 8, 8)
            # Policy: [N, 4096]
            p = torch.from_numpy(np_data[:, INPUT_SIZE : INPUT_SIZE + POLICY_SIZE])
            # Value: [N, 4]
            v = torch.from_numpy(np_data[:, INPUT_SIZE + POLICY_SIZE : ])

            # Prepend because we are loading Newest -> Oldest, but lists usually append.
            # Actually order in buffer doesn't matter for random sampling, 
            # so appending is fine.
            temp_states.append(s)
            temp_policies.append(p)
            temp_values.append(v)
            
            total_samples += num_samples_in_file
            loaded_files += 1

        if total_samples == 0:
            print("No data found.")
            return

        # Concatenate all loaded chunks
        self.states = torch.cat(temp_states, dim=0)
        self.policies = torch.cat(temp_policies, dim=0)
        self.values = torch.cat(temp_values, dim=0)

        # 4. Trim if we exceeded max_size (since we loaded whole files)
        # Since we loaded Newest first, we keep the beginning of the concatenated tensor
        # Wait, if we iterated Newest -> Oldest and appended, index 0 is Newest.
        if self.states.size(0) > self.max_size:
            self.states = self.states[:self.max_size]
            self.policies = self.policies[:self.max_size]
            self.values = self.values[:self.max_size]

        print(f"Replay Buffer Loaded: {self.states.size(0)} samples from {loaded_files} newest files.")

    def sample_batch(self, batch_size):
        # Random indices
        indices = torch.randint(0, self.states.size(0), (batch_size,))
        
        return (
            self.states[indices], 
            self.policies[indices], 
            self.values[indices]
        )

    def size(self):
        return self.states.size(0)

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Paths
    DATA_DIR = args.data_dir
    MODEL_SAVE_PATH = "model.pth"
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist.")
        return

    # 1. Load Sliding Window Buffer
    buffer = ReplayBuffer(DATA_DIR, args.max_buffer_size)
    if buffer.size() < args.batch_size:
        print("Not enough data to train. Waiting for more generation.")
        return

    # 2. Calculate Steps
    # Total positions generated in this specific C++ iteration
    new_samples = args.new_samples 
    
    # Formula: How many times do we want to see each new sample on average?
    # Total samples to process = new_samples * sampling_rate
    # Total batches = Total samples / batch_size
    
    total_samples_to_train = new_samples * args.sampling_rate
    num_steps = int(total_samples_to_train / args.batch_size)
    
    # Ensure at least 1 step if we have data and new samples > 0
    if num_steps == 0 and new_samples > 0:
        num_steps = 1
    
    print(f"New Samples: {new_samples}, Rate: {args.sampling_rate}, Batch: {args.batch_size}")
    print(f"Training for {num_steps} steps...")

    if num_steps == 0:
        print("Skipping training (0 steps calculated).")
        return

    # 3. Model Setup
    model = ChaturajiNN().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        except:
            print("Failed to load weights, starting fresh.")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 4. Training Loop (Step-based, not Epoch-based)
    total_loss = 0
    
    for step in range(num_steps):
        state, target_policy, target_value = buffer.sample_batch(args.batch_size)
        
        state = state.to(device)
        target_policy = target_policy.to(device)
        target_value = target_value.to(device)
        
        optimizer.zero_grad()
        
        pred_policy, pred_value = model(state)
        
        # Loss
        log_probs = F.log_softmax(pred_policy, dim=1)
        loss_policy = -torch.sum(target_policy * log_probs, dim=1).mean()
        loss_value = F.mse_loss(pred_value, target_value)
        
        loss = loss_policy + loss_value
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (step + 1) % 1 == 0:
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

    # 5. Export
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    export_to_onnx(MODEL_SAVE_PATH, "model.onnx")
    print("Training complete & model exported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--max-buffer-size", type=int, default=200000)
    parser.add_argument("--new-samples", type=int, default=0, help="Number of samples generated in this iteration")
    parser.add_argument("--sampling-rate", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    
    args = parser.parse_args()
    train_loop(args)