import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import glob
import argparse
import sys
import numpy as np

# Import model definitions
from model import (
    ChaturajiNN, export_to_onnx,
    NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_AREA,
    POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
)

# --- Configuration Constants ---
# Must match MAX_STORED_MOVES in types.h
MAX_STORED_MOVES = 64

# Define the numpy dtype that matches the C++ PackedSample struct exactly.
PACKED_DTYPE = np.dtype([
    ('piece_bbs',     np.uint64, (4, 5)), # [Player][PieceType]
    ('attack_bbs',    np.uint64, (4,)),   # [Player]
    ('points',        np.int32,  (4,)),   # [Player]
    ('full_move',     np.int32),
    ('last_reset',    np.int32),
    ('active_mask',   np.uint8),
    ('current_player',np.uint8),
    ('padding',       np.uint8,  (2,)),   # C++ struct padding for alignment
    ('num_policy',    np.int32),
    ('move_indices',  np.uint16, (MAX_STORED_MOVES,)),
    ('move_probs',    np.float32,(MAX_STORED_MOVES,)),
    ('values',        np.float32,(4,))
])

def unpack_batch_to_tensors(raw_batch):
    """
    Decompresses a batch of PackedSamples into PyTorch tensors.
    """
    batch_size = len(raw_batch)
    
    # --- 1. Expand Policy (Sparse -> Dense) ---
    policy_target = torch.zeros((batch_size, POLICY_OUTPUT_SIZE), dtype=torch.float32)
    legal_actions_mask = torch.zeros((batch_size, POLICY_OUTPUT_SIZE), dtype=torch.bool) # <--- NEW
    
    num_entries = raw_batch['num_policy']
    move_indices = raw_batch['move_indices'].astype(np.int64)
    move_probs = raw_batch['move_probs']

    # Vectorized Scatter
    row_indices = np.arange(batch_size)[:, None].repeat(MAX_STORED_MOVES, axis=1)
    col_indices = np.arange(MAX_STORED_MOVES)[None, :].repeat(batch_size, axis=0)
    mask = col_indices < num_entries[:, None]
    
    # Fill Probabilities
    policy_target[row_indices[mask], move_indices[mask]] = torch.from_numpy(move_probs[mask])
    
    # Fill Legal Mask (Any move present in MCTS data is considered legal)
    legal_actions_mask[row_indices[mask], move_indices[mask]] = True # <--- NEW

    # --- 2. Expand Values ---
    # The C++ struct stores Absolute values [Red, Blue, Yellow, Green].
    # The Network expects Relative values [Current, Next, Partner, Prev].
    
    abs_values = raw_batch['values']       # Shape (Batch, 4)
    cp_raw = raw_batch['current_player']   # Shape (Batch,)

    # Create indices: [[cp, cp+1, cp+2, cp+3], ...] % 4
    # Example: If cp=1 (Blue), indices are [1, 2, 3, 0]
    rel_indices = (cp_raw[:, None] + np.arange(4)) % 4

    # Gather values in relative order
    rel_values = np.take_along_axis(abs_values, rel_indices, axis=1)
    
    value_target = torch.from_numpy(rel_values)

    # --- 3. Expand Input State ---
    # We construct it as [Batch, 37, 64] first to match bitboard unpacking layout
    states_flat = torch.zeros((batch_size, NUM_INPUT_CHANNELS, BOARD_AREA), dtype=torch.float32)
    
    cp_raw = raw_batch['current_player']
    
    # Helper: Bitboard (Batch, N) -> Tensor (Batch, N, 64)
    def bbs_to_planes(bbs):
        view = bbs.view(np.uint8)
        bits = np.unpackbits(view, axis=-1, bitorder='little')
        t = torch.from_numpy(bits.astype(np.float32))
        # FIX: Explicitly reshape to separate the 'N' bitboards from the '64' squares
        return t.view(*bbs.shape, 64)

    # -- 3a. Piece Planes (Channels 0-19) --
    piece_bbs = raw_batch['piece_bbs']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        # Fancy indexing to gather specific player's boards for each batch item
        p_bbs = piece_bbs[np.arange(batch_size), abs_p, :] 
        planes = bbs_to_planes(p_bbs) # [Batch, 5, 64]
        states_flat[:, rel_i*5 : (rel_i+1)*5, :] = planes

    # -- 3b. Active Status (Channels 20-23) --
    active_mask = raw_batch['active_mask']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        is_active = (active_mask >> abs_p) & 1
        states_flat[:, 20 + rel_i, :] = torch.from_numpy(is_active[:, None].astype(np.float32))

    # -- 3c. Points (Channels 24-27) --
    points = raw_batch['points']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        p_pts = points[np.arange(batch_size), abs_p] / 100.0
        states_flat[:, 24 + rel_i, :] = torch.from_numpy(p_pts[:, None].astype(np.float32))

    # -- 3d. 50-Move Rule (Channel 28) --
    moves_since = raw_batch['full_move'] - raw_batch['last_reset']
    clock_val = np.clip(moves_since / 50.0, 0.0, 1.0)
    states_flat[:, 28, :] = torch.from_numpy(clock_val[:, None].astype(np.float32))

    # -- 3e. Attack Planes (Channels 29-32) --
    att_bbs = raw_batch['attack_bbs']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        p_att = att_bbs[np.arange(batch_size), abs_p][:, None]
        planes = bbs_to_planes(p_att).squeeze(1)
        states_flat[:, 29 + rel_i, :] = planes

    # -- 3f. In-Check Planes (Channels 33-36) --
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        king_bb = piece_bbs[np.arange(batch_size), abs_p, 4] 
        stressors = np.zeros(batch_size, dtype=np.uint64)
        for opp in range(4):
            is_enemy = (opp != abs_p)
            stressors |= np.where(is_enemy, att_bbs[:, opp], 0)
        in_check = (king_bb & stressors) != 0
        states_flat[:, 33 + rel_i, :] = torch.from_numpy(in_check[:, None].astype(np.float32))

    # --- 4. Spatial Rotation ---
    # Reshape to [Batch, 37, 8, 8]
    states = states_flat.view(batch_size, NUM_INPUT_CHANNELS, BOARD_DIM, BOARD_DIM)
    
    # Rotate based on Current Player to enforce relative perspective
    # Red (0): 0 deg, Blue (1): 90 CCW, Yellow (2): 180, Green (3): 270 CCW
    cp_torch = torch.from_numpy(cp_raw.astype(np.int64))
    
    for k in [1, 2, 3]:
        # Find indices in batch where rotation k is needed
        idx = (cp_torch == k).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            states[idx] = torch.rot90(states[idx], k=k, dims=[-2, -1])

    # Return the mask as the 4th value
    return states, policy_target, value_target, legal_actions_mask

class ReplayBuffer:
    def __init__(self, data_dir, max_size):
        self.data_dir = data_dir
        self.max_size = max_size
        self.data = None # This will hold the numpy structured array
        self.load_buffer()

    def load_buffer(self):
        files = glob.glob(os.path.join(self.data_dir, "gen_*.bin"))
        # Sort by modification time (newest first) to prioritize recent data
        files.sort(key=os.path.getmtime, reverse=True)
        
        chunks = []
        total_samples = 0
        
        print(f"[Python] Loading packed data from {len(files)} files...")
        
        for fp in files:
            if total_samples >= self.max_size: 
                break
            
            try:
                chunk = np.fromfile(fp, dtype=PACKED_DTYPE)
                
                if chunk.size == 0: continue
                
                chunks.append(chunk)
                total_samples += chunk.size
            except Exception as e:
                print(f"[Python] Warning: Skipping corrupt file {fp}: {e}")

        if total_samples > 0:
            # Concatenate chunks into one large array
            full_data = np.concatenate(chunks)
            
            # Trim to max size if exceeded
            if total_samples > self.max_size:
                full_data = full_data[:self.max_size]
                
            self.data = full_data
            
            # Print RAM usage statistics
            size_mb = self.data.nbytes / (1024 * 1024)
            print(f"[Python] Buffer loaded: {len(self.data)} samples.")
            print(f"[Python] RAM Usage: {size_mb:.2f} MB")
        else:
            print("[Python] No training data found.")
            self.data = np.array([], dtype=PACKED_DTYPE)

    def sample_batch(self, batch_size):
        """
        Randomly samples 'batch_size' items from the buffer and
        decompresses them into tensors.
        """
        if len(self.data) == 0: 
            return None
        
        # Random indices
        indices = np.random.randint(0, len(self.data), size=batch_size)
        
        # Get packed structs
        raw_batch = self.data[indices]
        
        # Decompress
        return unpack_batch_to_tensors(raw_batch)


def train_loop(args):
    # --- 1. Device Setup ---
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"[Python] Using Intel XPU: {torch.xpu.get_device_name(0)}")
        use_amp = False 
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[Python] Using CUDA")
        use_amp = True 
    else:
        device = torch.device("cpu")
        print("[Python] Using CPU")
        use_amp = False

    # --- 2. Paths ---
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_pth = os.path.join(args.save_dir, "latest.pth")
    opt_pth = os.path.join(args.save_dir, "latest.optimizer.pth")
    onnx_path = os.path.join(args.save_dir, "latest.onnx")

    # --- 3. Model & Optimizer ---
    model = ChaturajiNN().to(device)
    
    # Using NAdamW (Nesterov-accelerated Adam with Decoupled Weight Decay)
    optimizer = optim.NAdam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        decoupled_weight_decay=True
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"[Python] Optimizer: NadamW(lr={args.lr}, wd={args.wd})")

    # --- 4. Load Weights ---
    if args.load_weights:
        # If user provides .onnx, look for .pth
        target_pth = args.load_weights.replace(".onnx", ".pth") if args.load_weights.endswith(".onnx") else args.load_weights
        target_opt = target_pth.replace(".pth", ".optimizer.pth")

        if not os.path.exists(target_pth):
            print(f"[Python] FATAL ERROR: Weights file not found: {target_pth}")
            sys.exit(1)

        print(f"[Python] Loading model weights from {target_pth}")
        model.load_state_dict(torch.load(target_pth, map_location=device))

        if os.path.exists(target_opt):
            print(f"[Python] Loading optimizer state from {target_opt}")
            optimizer.load_state_dict(torch.load(target_opt, map_location=device))
            
            # Force command-line arguments to override saved optimizer state
            # (Allows changing LR/WD during training restarts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                param_group['weight_decay'] = args.wd
        else:
            print("[Python] No optimizer state found. Starting fresh optimizer.")
    else:
        print("[Python] No load-model specified. Initializing random weights.")

    # --- 5. Load Data ---
    buffer = ReplayBuffer(args.data_dir, args.max_buffer_size)
    
    # Check if we have enough data to train
    if len(buffer.data) < args.batch_size or args.new_samples == 0:
        print("[Python] Insufficient data or new_samples=0. Skipping training step.")
        return

    # Determine number of steps based on sampling rate
    # samples_to_train_on = new_generated_samples * sampling_rate
    num_steps = int((args.new_samples * args.sampling_rate) / args.batch_size)
    if num_steps == 0:
        print("[Python] Sampling rate results in 0 steps. Skipping.")
        return

    print(f"[Python] Training for {num_steps} steps (Batch: {args.batch_size})...")

    # --- 6. Training Loop ---
    model.train()
    
    for step in range(num_steps):
        batch = buffer.sample_batch(args.batch_size)
        if batch is None: break
        
        # Unpack the mask as well
        s, tp, tv, mask = batch
        s, tp, tv, mask = s.to(device), tp.to(device), tv.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            p, v = model(s)

            # AlphaZero Loss:
            # 1. Policy: Cross Entropy (maximize log prob of target)
            # 2. Value: MSE
            
            # --- MASKING ILLEGAL MOVES ---
            # Set logits for illegal moves to negative infinity (-1e9).
            # When passed to log_softmax, these become 0 probability.
            # This prevents the network from "learning" to push them down endlessly.
            p_masked = torch.where(mask, p, torch.tensor(-1e9, device=device))

            loss_policy = -torch.sum(tp * F.log_softmax(p_masked, dim=1), dim=1).mean()
            loss_value = F.mse_loss(v, tv)
            
            loss = loss_policy + 4 * loss_value
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Simple logging
        print(f"  Step {step+1}/{num_steps} | Loss: {loss.item():.4f} (Pol: {loss_policy.item():.4f}, Val: {loss_value.item():.4f})")

    # --- 7. Save & Export ---
    print(f"[Python] Saving checkpoint to {model_pth}")
    torch.save(model.state_dict(), model_pth)
    torch.save(optimizer.state_dict(), opt_pth)
    
    print(f"[Python] Exporting to ONNX...")
    export_to_onnx(model_pth, onnx_path)
    print("[Python] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--max-buffer-size", type=int, default=1000000)
    parser.add_argument("--new-samples", type=int, default=0)
    parser.add_argument("--sampling-rate", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--load-weights", type=str, default="")
    
    args = parser.parse_args()
    
    try:
        train_loop(args)
    except KeyboardInterrupt:
        print("[Python] Training interrupted.")
    except Exception as e:
        print(f"[Python] Exception during training: {e}")
        # Re-raise to ensure C++ knows something went wrong
        raise e