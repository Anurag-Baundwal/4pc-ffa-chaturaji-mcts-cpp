import os
import sys
import time
import threading
import argparse
import json
import subprocess
import shlex
import platform
import numpy as np
import pyautogui

# Win32 imports for robust window switching on Windows
try:
    import win32gui
    import win32con
    import win32process
    import win32api
except ImportError:
    print("Error: 'pywin32' is required. Install via: pip install pywin32")
    sys.exit(1)

from board_locator import BoardLocator

# --- CONFIGURATION ---
WINDOW_TITLE_TARGET = "Account-1"  # The browser window title to look for
GAME_STATE_FILE = "game_state.json"
POLL_INTERVAL_SECONDS = 0.1
PIECE_ASSET_PATH = os.path.join('assets', 'pieces') 
ENGINE_BINARY = os.path.join("bazel-bin", "chaturaji_engine.exe") # Adjust if needed

# -----------------------------------------------------------------------------

def launch_move_fetcher(url: str):
    """Launches move_fetcher.py in a separate terminal."""
    current_os = platform.system()
    print(f"[{current_os}] Launching move fetcher...")
    fetcher_args = [sys.executable, "move_fetcher.py", "--url", url]
    
    try:
        if current_os == "Windows":
            subprocess.Popen(['start', 'cmd', '/k'] + fetcher_args, shell=True)
        elif current_os == "Linux":
            subprocess.Popen(['gnome-terminal', '--'] + fetcher_args)
        else: # macOS or fallback
            print(f"Please run manually: {' '.join(fetcher_args)}")
    except Exception as e:
        print(f"Error launching fetcher: {e}")

# --- WINDOW HELPERS ---

class WindowFinder:
    def __init__(self, partial_title: str):
        self._hwnd = 0
        self.partial_title = partial_title.lower()

    def callback(self, hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            text = win32gui.GetWindowText(hwnd).lower()
            if self.partial_title in text:
                self._hwnd = hwnd

    @property
    def hwnd(self):
        return self._hwnd

def switch_to_window(partial_title: str) -> bool:
    """Activates the game window using Win32 APIs."""
    finder = WindowFinder(partial_title)
    win32gui.EnumWindows(finder.callback, None)
    hwnd = finder.hwnd
    
    if not hwnd: return False

    try:
        fg_hwnd = win32gui.GetForegroundWindow()
        if fg_hwnd == hwnd: return True

        # Force foreground (Windows 10/11 restriction workaround)
        fg_tid, _ = win32process.GetWindowThreadProcessId(fg_hwnd)
        cur_tid = win32api.GetCurrentThreadId()
        win32process.AttachThreadInput(cur_tid, fg_tid, True)
        
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        win32process.AttachThreadInput(cur_tid, fg_tid, False)
        
        time.sleep(0.05)
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False

# --- ENGINE WRAPPER ---

class EngineHandler:
    """
    Manages the C++ Engine process.
    Assumes the engine accepts commands via stdin line-by-line.
    """
    def __init__(self, binary_path, model_path, sims, batch_size):
        self.binary_path = binary_path
        self.model_path = model_path
        self.sims = sims
        self.batch_size = batch_size
        self.process = None
        self._start_process()

    def _start_process(self):
        """Starts the engine process with specific flags."""
        if not os.path.exists(self.binary_path):
            print(f"CRITICAL: Engine binary not found at {self.binary_path}")
            sys.exit(1)

        cmd = [
            self.binary_path, 
            "--model", self.model_path, 
            "--interactive",
            "--mcts-batch", str(self.batch_size)
        ]  
        
        print(f"[ENGINE] Starting: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def get_turn_color(self, moves_list):
        """
        Updates the engine's internal board and queries whose turn it is.
        Returns: 'r', 'b', 'y', 'g', or None (game over/error).
        """
        if self.process.poll() is not None:
            print("[ENGINE] Process died. Restarting...")
            self._start_process()

        # 1. Update Position
        moves_str = " ".join(moves_list)
        self._send(f"position moves {moves_str}")

        # 2. Query Turn
        self._send("turn")

        # 3. Read response
        while True:
            try:
                line = self.process.stdout.readline()
                if not line: break
                line = line.strip()
                
                if line.startswith("turn"):
                    # Output format: "turn r" or "turn gameover"
                    parts = line.split()
                    if len(parts) > 1:
                        color = parts[1]
                        if color == "gameover":
                            return None
                        return color # 'r', 'b', 'y', 'g'
                    return None
            except Exception as e:
                print(f"[ENGINE] Error reading turn: {e}")
                return None
        return None

    def search(self, moves_list):
        """
        Sends current history to engine and waits for 'bestmove'.
        Protocol assumed:
          > position moves <m1> <m2> ...
          > go sims <N>
          < ... info ...
          < bestmove <move>
        """
        if self.process.poll() is not None:
            print("[ENGINE] Process died. Restarting...")
            self._start_process()

        # Construct position command
        moves_str = " ".join(moves_list)
        self._send(f"position moves {moves_str}")
        
        # Construct go command
        self._send(f"go sims {self.sims}")

        # Read output until bestmove
        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            
            # Optional: Print info lines to see search progress
            if line.startswith("info") or line.startswith("root"):
                print(f"[ENGINE] {line}")
            
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
                return None
        return None

    def _send(self, cmd):
        if self.process:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def stop(self):
        if self.process:
            self.process.terminate()

# --- CONTROLLER ---

class GameController:
    def __init__(self, my_color: str, model_path: str, sims: int, batch_size: int):
        self.my_color = my_color.upper() # R, B, Y, or G
        self.board_moves = []
        self.geometry = None
        
        # Initialize engine
        self.engine = EngineHandler(ENGINE_BINARY, model_path, sims, batch_size)

        
        # Calibrate Board
        self._calibrate_board()

        # File Polling
        self.last_sync_time = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll_file, daemon=True)
        self.thread.start()

    def _calibrate_board(self):
        print(f"\n[INIT] Looking for window: '{WINDOW_TITLE_TARGET}' for calibration...")
        
        # Retry finding window
        found = False
        for _ in range(5):
            if switch_to_window(WINDOW_TITLE_TARGET):
                found = True
                break
            time.sleep(1)
            
        if not found:
            print("CRITICAL: Game window not found. Ensure browser is open and title matches.")
            sys.exit(1)

        locator = BoardLocator(PIECE_ASSET_PATH, WINDOW_TITLE_TARGET)
        self.geometry = locator.calibrate()
        
        if not self.geometry:
            print("CRITICAL: Board calibration failed. Make sure board is visible.")
            sys.exit(1)
        
        print("[INIT] Calibration successful.")

    def _poll_file(self):
        print("[POLLER] Listening for game updates...")
        while not self.stop_event.is_set():
            try:
                if os.path.exists(GAME_STATE_FILE):
                    mtime = os.path.getmtime(GAME_STATE_FILE)
                    if mtime > self.last_sync_time:
                        time.sleep(0.05) # Tiny buffer for write completion
                        with open(GAME_STATE_FILE, 'r') as f:
                            data = json.load(f)
                        
                        if data['detection_timestamp'] > self.last_sync_time:
                            self.last_sync_time = data['detection_timestamp']
                            self._on_new_state(data['moves'])
            except Exception as e:
                print(f"Error reading state: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)

    def _on_new_state(self, moves):
        """Called when move fetcher detects a new board state."""
        self.board_moves = moves
        ply = len(moves)
        turn_color = self.engine.get_turn_color(moves)
        
        if turn_color is None:
            print(f"\n[STATE] Ply: {ply} | Game Over")
            return

        turn_color = turn_color.upper()
        
        print(f"\n[STATE] Ply: {ply} | Turn: {turn_color}")
        
        if turn_color == self.my_color:
            print(f"--- MY TURN ({self.my_color}) ---")
            self._play_move()

    def _play_move(self):
        # 1. Ask Engine
        best_move = self.engine.search(self.board_moves)
        
        if best_move:
            print(f"[PLAY] Engine chose: {best_move}")
            self._execute_gui_move(best_move)
        else:
            print("[PLAY] Engine returned no move (Game Over or Error).")

    def _execute_gui_move(self, move_str):
        """Converts e.g. 'a1a2' to pixels and drags."""
        if not switch_to_window(WINDOW_TITLE_TARGET):
            print("Error: Could not focus window to move.")
            return

        try:
            # Parse move string (e.g., 'e2e4' or 'a7a8r')
            # src_sq is always characters 0 and 1
            # dst_sq is always characters 2 and 3
            src_sq = move_str[0:2]
            dst_sq = move_str[2:4]

            print(f"[GUI] Executing move: {src_sq} to {dst_sq}")

            src_px = self._alg_to_px(src_sq)
            dst_px = self._alg_to_px(dst_sq)

            # Play the move (drag and drop)
            pyautogui.moveTo(src_px[0], src_px[1])
            pyautogui.click() 
            time.sleep(0.05)
            pyautogui.dragTo(dst_px[0], dst_px[1], button='left', duration=0.1)
            
        except Exception as e:
            print(f"Error executing GUI move: {e}")

    def _alg_to_px(self, alg):
        """
        Converts 'a1' to (x, y) pixels.
        Logic: Origin is a1 center.
        Pixel = Origin + (col * u_col) + (row * u_row)
        """
        col_idx = ord(alg[0]) - ord('a') # 0..7
        row_idx = int(alg[1]) - 1        # 0..7
        
        origin = self.geometry['origin_a1_center_px']
        u_col = self.geometry['unit_vec_col']
        u_row = self.geometry['unit_vec_row']
        
        px = origin + (col_idx * u_col) + (row_idx * u_row)
        return int(px[0]), int(px[1])

    def shutdown(self):
        self.stop_event.set()
        self.engine.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', required=True, choices=['r','b','y','g'], help="Your color (r, b, y, or g)")
    parser.add_argument('--url', required=True, help="Chess.com game URL")
    parser.add_argument('--model', required=True, help="Path to ONNX model")
    parser.add_argument('--sims', type=int, default=35000, help="Simulations per move")
    parser.add_argument('--batch', type=int, default=64, help="MCTS Batch Size")

    
    args = parser.parse_args()

    # Reset game state file
    with open(GAME_STATE_FILE, 'w') as f:
        json.dump({'moves': [], 'detection_timestamp': 0}, f)

    # Launch Fetcher
    launch_move_fetcher(args.url)
    
    # Wait for fetcher to init
    time.sleep(3)

    # Start Controller
    controller = GameController(args.color, args.model, args.sims, args.batch)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        controller.shutdown()