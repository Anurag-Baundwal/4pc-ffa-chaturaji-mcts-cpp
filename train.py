import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import struct
import os
import glob
from model import ChaturajiNN, export_for_cpp

# Data dimensions
INPUT_SIZE = 33 * 8 * 8
POLICY_SIZE = 4096
VALUE_SIZE = 4
SAMPLE_SIZE_BYTES = (INPUT_SIZE + POLICY_SIZE + VALUE_SIZE) * 4 # 4 bytes per float

class ChaturajiBinaryDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = glob.glob(os.path.join(data_dir, "*.bin"))
        self.data = []
        
        print(f"Loading data from {len(self.data_files)} files...")
        for filepath in self.data_files:
            with open(filepath, "rb") as f:
                content = f.read()
                num_samples = len(content) // SAMPLE_SIZE_BYTES
                if len(content) % SAMPLE_SIZE_BYTES != 0:
                    print(f"Warning: File {filepath} has incomplete samples.")
                
                # We load everything into memory for speed. 
                # For massive datasets, you'd use mmap or lazy loading.
                self.data.append(content)
        
        self.file_indices = []
        total_samples = 0
        for i, content in enumerate(self.data):
            count = len(content) // SAMPLE_SIZE_BYTES
            self.file_indices.append((total_samples, total_samples + count, i))
            total_samples += count
        self.total_samples = total_samples
        print(f"Total samples loaded: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which file chunk contains this index
        # (Simple linear search is fine for small number of files, optimize if needed)
        chunk_idx = 0
        local_idx = idx
        for start, end, c_idx in self.file_indices:
            if idx >= start and idx < end:
                chunk_idx = c_idx
                local_idx = idx - start
                break
        
        offset = local_idx * SAMPLE_SIZE_BYTES
        raw_bytes = self.data[chunk_idx][offset : offset + SAMPLE_SIZE_BYTES]
        
        # Unpack
        floats = struct.unpack(f'{INPUT_SIZE + POLICY_SIZE + VALUE_SIZE}f', raw_bytes)
        
        # Reshape Input [33, 8, 8]
        state = torch.tensor(floats[:INPUT_SIZE]).view(33, 8, 8)
        
        # Policy
        policy = torch.tensor(floats[INPUT_SIZE : INPUT_SIZE + POLICY_SIZE])
        
        # Value
        value = torch.tensor(floats[INPUT_SIZE + POLICY_SIZE:])
        
        return state, policy, value

def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Paths
    DATA_DIR = "./training_data" # Where C++ writes the .bin files
    MODEL_SAVE_PATH = "model.pth"
    JIT_EXPORT_PATH = "model.pt" # Compatible with C++
    
    # Hyperparams
    BATCH_SIZE = 512
    EPOCHS = 1
    LR = 0.001
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Run C++ self-play to generate data.")
        return

    dataset = ChaturajiBinaryDataset(DATA_DIR)
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    model = ChaturajiNN().to(device)
    
    # Load existing weights if available
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Loaded existing model weights.")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (state, target_policy, target_value) in enumerate(dataloader):
            state = state.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            
            optimizer.zero_grad()
            
            pred_policy, pred_value = model(state)
            
            # Loss Calculation
            # Policy: Cross Entropy (target_policy is probability distribution)
            # We use LogSoftmax + manual sum because target is soft probabilities, not indices
            log_probs = F.log_softmax(pred_policy, dim=1)
            loss_policy = -torch.sum(target_policy * log_probs, dim=1).mean()
            
            # Value: MSE
            loss_value = F.mse_loss(pred_value, target_value)
            
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f} (P: {loss_policy.item():.4f}, V: {loss_value.item():.4f})")

    # Save Python weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Export for C++
    export_for_cpp(MODEL_SAVE_PATH, JIT_EXPORT_PATH)
    print("Training complete. Model exported.")

if __name__ == "__main__":
    train_loop()