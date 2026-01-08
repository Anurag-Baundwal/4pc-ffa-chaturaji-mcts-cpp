import time
import pyautogui
from board_locator import BoardLocator

def test_calibration():
    # Path to where you saved your rook PNGs
    ASSET_PATH = "assets/pieces" 
    
    # You can put anything for window_title for now as the script 
    # currently just takes a full screenshot.
    locator = BoardLocator(piece_assets_path=ASSET_PATH, window_title="Chess")

    print("--- Starting Calibration Test ---")
    print("Please make sure the Chaturaji board is visible on your screen.")
    print("Starting in 3 seconds...")
    time.sleep(3)

    geometry = locator.calibrate()

    if geometry:
        print("\nSUCCESS! Geometry calculated.")
        print(f"Square Size: {geometry['square_size']:.2f}px")
        
        # Test 1: Move mouse to the four corners
        corners = {
            "a1 (Red)": (0, 0),
            "a8 (Blue)": (0, 7),
            "h8 (Yellow)": (7, 7),
            "h1 (Green)": (7, 0)
        }

        print("\nMoving mouse to corners to verify...")
        origin = geometry['origin_a1_center_px']
        u_col = geometry['unit_vec_col']
        u_row = geometry['unit_vec_row']

        for name, (col, row) in corners.items():
            # Calculate pixel: Origin + (col * width) + (row * height)
            target_px = origin + (col * u_col) + (row * u_row)
            
            print(f"Moving to {name}...")
            pyautogui.moveTo(target_px[0], target_px[1], duration=0.1)
            time.sleep(0.5)

        print("\nCalibration looks correct if the mouse landed in the center of the corner pieces.")
    else:
        print("\nFAILED: Could not calibrate. Check your rook PNGs and ensure the board is visible.")

if __name__ == "__main__":
    test_calibration()