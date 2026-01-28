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
    ('piece_bbs',     np.uint64, (4, 5)), 
    ('attack_bbs',    np.uint64, (4,)),   
    ('xray_attack_bbs', np.uint64, (4,)),
    
    # Hand-Crafted Heuristic Scalars 
    ('material',      np.float32, (4,)),
    ('pawn_count',    np.float32, (4,)),
    ('avg_pawn_dist', np.float32, (4,)),
    ('king_safe',     np.float32, (4,)),
    ('pawns_conn',    np.float32, (4,)),

    # Game State Scalars
    ('points',        np.int32,  (4,)),   
    ('full_move',     np.int32),
    ('last_reset',    np.int32),
    ('active_mask',   np.uint8),
    ('current_player',np.uint8),
    
    # Padding for alignment
    ('padding',       np.uint8,  (2,)),   

    # Policy & Value
    ('num_policy',    np.int32),
    ('move_indices',  np.uint16, (MAX_STORED_MOVES,)),
    ('move_probs',    np.float32,(MAX_STORED_MOVES,)),
    ('values',        np.float32,(4,))
])

def unpack_batch_to_tensors(raw_batch):
    """
    Decompresses a batch of PackedSamples into PyTorch tensors.
    
    Channel Map (Total 62):
    0-19:  Piece Bitboards (5 types * 4 players)
    20-23: Material Score (Scalar)
    24-27: Pawn Count (Scalar)
    28-31: Connected Pawns Flag (Scalar)
    32-35: Avg Pawn Distance to King (Scalar)
    36-39: King Safe Moves (Scalar)
    40-43: X-Ray Attack Bitboards
    44-47: Active Status
    48-51: Points
    52:    50-Move Clock
    53-56: Standard Attack Bitboards
    57-60: In-Check Flags
    61:    Active Opponent Count (Scalar)
    """
    batch_size = len(raw_batch)
    
    # --- 1. Expand Policy (Sparse -> Dense) ---
    policy_target = torch.zeros((batch_size, POLICY_OUTPUT_SIZE), dtype=torch.float32)
    legal_actions_mask = torch.zeros((batch_size, POLICY_OUTPUT_SIZE), dtype=torch.bool)
    
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
    legal_actions_mask[row_indices[mask], move_indices[mask]] = True

    # --- 2. Expand Values ---
    # The C++ struct stores Absolute values [Red, Blue, Yellow, Green].
    # The Network expects Relative values [Current, Next, Partner, Prev].
    
    abs_values = raw_batch['values']       # Shape (Batch, 4)
    cp_raw = raw_batch['current_player']   # Shape (Batch,)

    # Create indices: [[cp, cp+1, cp+2, cp+3], ...] % 4
    rel_indices = (cp_raw[:, None] + np.arange(4)) % 4

    # Gather values in relative order
    rel_values = np.take_along_axis(abs_values, rel_indices, axis=1)
    value_target = torch.from_numpy(rel_values)

    # --- 3. Expand Input State ---
    # Shape: [Batch, 62, 64]
    states_flat = torch.zeros((batch_size, NUM_INPUT_CHANNELS, BOARD_AREA), dtype=torch.float32)
    
    # Helper: Bitboard (Batch, N) -> Tensor (Batch, N, 64)
    def bbs_to_planes(bbs):
        view = bbs.view(np.uint8)
        bits = np.unpackbits(view, axis=-1, bitorder='little')
        t = torch.from_numpy(bits.astype(np.float32))
        return t.view(*bbs.shape, 64)

    # -- 3a. Piece Planes (Channels 0-19) --
    piece_bbs = raw_batch['piece_bbs']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        p_bbs = piece_bbs[np.arange(batch_size), abs_p, :] 
        planes = bbs_to_planes(p_bbs) # [Batch, 5, 64]
        states_flat[:, rel_i*5 : (rel_i+1)*5, :] = planes

    # -- 3b. New Hand-Crafted Heuristics (Channels 20-43) --
    
    # Material (20-23)
    material = raw_batch['material']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        val = material[np.arange(batch_size), abs_p]
        states_flat[:, 20 + rel_i, :] = torch.from_numpy(val[:, None].astype(np.float32))

    # Pawn Count (24-27)
    pawn_cnt = raw_batch['pawn_count']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        val = pawn_cnt[np.arange(batch_size), abs_p]
        states_flat[:, 24 + rel_i, :] = torch.from_numpy(val[:, None].astype(np.float32))

    # Connected Pawns Flag (28-31)
    pawns_conn = raw_batch['pawns_conn']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        val = pawns_conn[np.arange(batch_size), abs_p]
        states_flat[:, 28 + rel_i, :] = torch.from_numpy(val[:, None].astype(np.float32))

    # Avg Pawn Distance (32-35)
    avg_dist = raw_batch['avg_pawn_dist']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        val = avg_dist[np.arange(batch_size), abs_p]
        states_flat[:, 32 + rel_i, :] = torch.from_numpy(val[:, None].astype(np.float32))

    # King Safe Moves (36-39)
    king_safe = raw_batch['king_safe']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        val = king_safe[np.arange(batch_size), abs_p]
        states_flat[:, 36 + rel_i, :] = torch.from_numpy(val[:, None].astype(np.float32))

    # X-Ray Attacks (40-43)
    xray_bbs = raw_batch['xray_attack_bbs']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        states_flat[:, 40 + rel_i, :] = bbs_to_planes(xray_bbs[np.arange(batch_size), abs_p])

    # -- 3c. Original Metadata (Channels 44-60) --

    # Active Status (44-47)
    active_mask = raw_batch['active_mask']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        is_active = (active_mask >> abs_p) & 1
        states_flat[:, 44 + rel_i, :] = torch.from_numpy(is_active[:, None].astype(np.float32))

    # Points (48-51)
    points = raw_batch['points']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        p_pts = points[np.arange(batch_size), abs_p] / 50.0
        states_flat[:, 48 + rel_i, :] = torch.from_numpy(p_pts[:, None].astype(np.float32))

    # 50-Move Rule (52)
    moves_since = raw_batch['full_move'] - raw_batch['last_reset']
    clock_val = np.clip(moves_since / 50.0, 0.0, 1.0)
    states_flat[:, 52, :] = torch.from_numpy(clock_val[:, None].astype(np.float32))

    # Standard Attack Planes (53-56)
    att_bbs = raw_batch['attack_bbs']
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        states_flat[:, 53 + rel_i, :] = bbs_to_planes(att_bbs[np.arange(batch_size), abs_p])

    # In-Check Planes (57-60)
    for rel_i in range(4):
        abs_p = (cp_raw + rel_i) % 4
        king_bb = piece_bbs[np.arange(batch_size), abs_p, 4] 
        stressors = np.zeros(batch_size, dtype=np.uint64)
        for opp in range(4):
            # Combined attacks of all enemies
            is_enemy = (opp != abs_p)
            stressors |= np.where(is_enemy, att_bbs[:, opp], 0)
        in_check = (king_bb & stressors) != 0
        states_flat[:, 57 + rel_i, :] = torch.from_numpy(in_check[:, None].astype(np.float32))

    # Active Opponent Count (61)
    # Count bits in active_mask using vectorized popcount
    total_active = np.array([bin(m).count('1') for m in active_mask])
    opp_count_val = (total_active - 1) / 3.0
    states_flat[:, 61, :] = torch.from_numpy(opp_count_val[:, None].astype(np.float32))
    
    # --- 4. Spatial Rotation ---
    # Reshape to [Batch, 62, 8, 8]
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

    print(f"[Python] Optimizer: NadamW(lr={args.lr}, wd={args.wd}) | Uncertainty Weighting: Enabled")

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
    
    # Use a safe mask value for FP16 if AMP is active (max for fp16 is ~65k)
    MASK_VALUE = -30000.0 if use_amp else -1e8
    
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
            # Apply a large negative mask to illegal move logits. This ensures that the 
            # subsequent softmax operation assigns near-zero probability to these moves, 
            # concentrating the network's predictive mass on the legal action space.
            p_masked = torch.where(mask, p, torch.full_like(p, MASK_VALUE))
            
            # Calculate the log-softmax of the masked logits.
            log_p = F.log_softmax(p_masked, dim=1)

            # Zero out the log-probabilities of illegal moves before the loss calculation.
            # This prevents numerical 'NaN' errors that occur when a zero target probability 
            # is multiplied by a negative infinity log-probability.
            log_p_safe = torch.where(mask, log_p, torch.zeros_like(log_p))
            
            # Policy uses Cross-Entropy (Negative Log Likelihood)
            loss_policy_raw = -torch.sum(tp * log_p_safe, dim=1).mean()
            # Value uses Mean Squared Error
            loss_value_raw = F.mse_loss(v, tv)

            # --- DYNAMIC LOSS WEIGHTING (Uncertainty Weighting) ---
            # Multi-Task Learning using Uncertainty (Kendall et al.)
            # L = [ (1/sigma_p^2) * L_p + log(sigma_p) ] + [ (1/2*sigma_v^2) * L_v + log(sigma_v) ]
            # We let s = log(sigma^2), so exp(-s) = 1/sigma^2.
            
            s_p = model.log_vars[0] # Policy log-variance
            s_v = model.log_vars[1] # Value log-variance

            # Policy weighting (Classification formulation)
            weighted_loss_p = torch.exp(-s_p) * loss_policy_raw + 0.5 * s_p
            # Value weighting (Regression formulation)
            weighted_loss_v = 0.5 * torch.exp(-s_v) * loss_value_raw + 0.5 * s_v

            loss = weighted_loss_p + weighted_loss_v
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- PREVENT TASK NEGLECT ---
        # Manually clamp the learnable log_vars to a reasonable range.
        # This prevents the optimizer from "muting" a head by allowing its
        # uncertainty to grow toward infinity.
        with torch.no_grad():
            model.log_vars.clamp_(-5.0, 5.0)
        
        # Calculate actual weights for logging
        w_p = torch.exp(-s_p).item()
        w_v = (0.5 * torch.exp(-s_v)).item()
        
        lp_val = loss_policy_raw.item()
        lv_val = loss_value_raw.item()
        
        # Logging
        print(f"  Step {step+1}/{num_steps} | Loss: {loss.item():.4f} "
              f"| Raw (Pol: {lp_val:.4f}, Val: {lv_val:.4f}) "
              f"| Weights (P_w: {w_p:.2f}, V_w: {w_v:.2f})")

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