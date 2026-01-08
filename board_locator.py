import os
import time
import cv2
import numpy as np
import pyautogui
from typing import Dict, Tuple, Optional

class BoardLocator:
    """
    Finds the 8x8 Chaturaji board on screen by locating the Rooks (Boats) in the corners.
    
    Corner Mapping:
    - Red Rook    -> a1
    - Blue Rook   -> a8
    - Yellow Rook -> h8
    - Green Rook  -> h1
    
    Calibration calculates the unit vectors for moving 1 square right (col) and 1 square up (row)
    based on the actual positions of the rooks, handling 0, 90, 180, and 270 degree rotations.
    """
    def __init__(self, piece_assets_path: str, window_title: str, confidence: float = 0.85):
        self.piece_path = piece_assets_path
        self.window_title = window_title 
        self.confidence = confidence
        self.rook_templates = self._load_rook_templates()

    def _load_rook_templates(self) -> Dict[str, np.ndarray]:
        """
        Loads the four colored rook PNGs into OpenCV format.
        Expected filenames: red_rook.png, blue_rook.png, etc.
        """
        templates = {}
        # Maps internal color code to filename prefix
        color_map = {
            'R': 'red_rook.png',
            'B': 'blue_rook.png',
            'Y': 'yellow_rook.png',
            'G': 'green_rook.png',
        }
        
        print(f"Loading rook templates from {self.piece_path}...")
        for color_code, filename in color_map.items():
            full_path = os.path.join(self.piece_path, filename)

            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Rook template not found at {full_path}")

            # Load images with alpha channel (IMREAD_UNCHANGED)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Could not read image file for {color_code} Rook.")
            
            templates[color_code] = img
            print(f"  - Loaded {color_code} Rook template.")
        return templates

    def _find_piece_center(self, screenshot: np.ndarray, template: np.ndarray) -> Optional[Tuple[int, int]]:
        """Finds the center of a given piece template within a screenshot."""
        # Ensure template has an alpha channel for masking
        if template.shape[2] < 4:
            # Fallback to simple matching if no alpha
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        else:
            # Use alpha mask for precise matching (ignores background color of square)
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            template_bgr = template[:, :, :3]
            alpha_mask = template[:, :, 3]
            res = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=alpha_mask)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= self.confidence:
            h, w = template.shape[:2]
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
        return None

    def calibrate(self) -> Optional[Dict]:
        """
        Main method. Finds Rooks on screen to calculate board geometry.
        Requires finding at least Red and Yellow rooks to determine the board bounds.
        """
        print("Taking a screenshot of the entire screen...")
        try:
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGRA)
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None

        print("Searching for Rooks on screen...")
        rook_coords = {}
        
        # 1. Locate all 4 Rooks
        for color, template in self.rook_templates.items():
            coords = self._find_piece_center(screenshot_cv, template)
            if coords:
                rook_coords[color] = np.array(coords, dtype=float)
                print(f"  - Found {color} Rook at {coords}")
            else:
                print(f"  - Could not find {color} Rook.")

        # 2. Validation: We specifically need Red (a1) to serve as origin.
        if 'R' not in rook_coords:
            print("!!! CRITICAL: Failed to find Red rook (a1 origin).")
            return None

        return self._calculate_geometry(rook_coords)

    def _calculate_geometry(self, rook_coords: Dict[str, np.ndarray]) -> Optional[Dict]:
        """
        Calculates grid vectors.
        Logic:
        - a1->h1 (Red->Green) defines the Column Vector (u_col).
        - a1->a8 (Red->Blue) defines the Row Vector (u_row).
        
        If Green or Blue are missing, we deduce them from Yellow using the
        geometric property that the board is a square grid.
        """
        pos_red = rook_coords['R']
        
        # --- Attempt 1: Direct Calculation from edges (Most Accurate) ---
        if 'G' in rook_coords and 'B' in rook_coords:
            # Red to Green is 7 columns
            unit_vec_col = (rook_coords['G'] - pos_red) / 7.0
            # Red to Blue is 7 rows
            unit_vec_row = (rook_coords['B'] - pos_red) / 7.0
            print("Method: Direct edge calculation (Found R, G, B).")

        # --- Attempt 2: Diagonal Deduction (if an edge is blocked/missing) ---
        elif 'Y' in rook_coords:
            print("Method: Diagonal deduction (Found R, Y, and maybe one other).")
            # vec_diag = Yellow - Red
            vec_diag = rook_coords['Y'] - pos_red
            dx, dy = vec_diag[0], vec_diag[1]
            
            # Use signs of diagonal to determine rotation/orientation
            # This assumes the board is roughly axis-aligned on screen (0, 90, 180, 270 deg)
            
            if dx > 0 and dy < 0:
                # Rot 0 (Red Bottom): a1=BL, h8=TR -> Diag (+, -)
                # Col is (+, 0), Row is (0, -)
                print("Detected Orientation: Standard (Red at Bottom)")
                unit_vec_col = np.array([abs(dx)/7.0, 0.0])
                unit_vec_row = np.array([0.0, -abs(dy)/7.0])
                
            elif dx < 0 and dy < 0:
                # Rot 90 (Blue Bottom): a1=BR, h8=TL -> Diag (-, -)
                # Col is (0, -), Row is (-, 0)
                print("Detected Orientation: 90 deg CCW (Blue at Bottom)")
                unit_vec_col = np.array([0.0, -abs(dy)/7.0])
                unit_vec_row = np.array([-abs(dx)/7.0, 0.0])
                
            elif dx < 0 and dy > 0:
                # Rot 180 (Yellow Bottom): a1=TR, h8=BL -> Diag (-, +)
                # Col is (-, 0), Row is (0, +)
                print("Detected Orientation: 180 deg (Yellow at Bottom)")
                unit_vec_col = np.array([-abs(dx)/7.0, 0.0])
                unit_vec_row = np.array([0.0, abs(dy)/7.0])
                
            else: # dx > 0, dy > 0
                # Rot 270 (Green Bottom): a1=TL, h8=BR -> Diag (+, +)
                # Col is (0, +), Row is (+, 0)
                print("Detected Orientation: 270 deg CCW (Green at Bottom)")
                unit_vec_col = np.array([0.0, abs(dy)/7.0])
                unit_vec_row = np.array([abs(dx)/7.0, 0.0])
                
            # If we have one edge piece, overwrite that specific vector for better accuracy
            if 'G' in rook_coords:
                unit_vec_col = (rook_coords['G'] - pos_red) / 7.0
            if 'B' in rook_coords:
                unit_vec_row = (rook_coords['B'] - pos_red) / 7.0

        else:
            print("!!! CRITICAL: Not enough rooks found. Need at least (R, G, B) OR (R, Y).")
            return None

        square_size = (np.linalg.norm(unit_vec_col) + np.linalg.norm(unit_vec_row)) / 2.0
        print(f"Calculated average square size: {square_size:.2f} pixels")
        
        # Origin is the center of a1 (Red Rook)
        origin_a1_center_px = pos_red
        
        # --- Sanity Checks ---
        # 1. Check Yellow (h8) position if we computed from edges
        if 'Y' in rook_coords:
            expected_yellow = origin_a1_center_px + (7 * unit_vec_col) + (7 * unit_vec_row)
            error = np.linalg.norm(rook_coords['Y'] - expected_yellow)
            if error > square_size / 2:
                print(f"WARNING: Yellow rook position mismatch! Error: {error:.2f}px")

        print(f"Unit Vector Col (a->b): {unit_vec_col.round(2)}")
        print(f"Unit Vector Row (1->2): {unit_vec_row.round(2)}")
        print(f"Origin (a1): {origin_a1_center_px.round(1)}")
        
        return {
            'origin_a1_center_px': origin_a1_center_px,
            'unit_vec_col': unit_vec_col,
            'unit_vec_row': unit_vec_row,
            'square_size': square_size
        }