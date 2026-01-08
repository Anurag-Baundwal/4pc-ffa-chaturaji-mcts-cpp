import time
import random
import os
import argparse
import sys
import pickle
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION ---
GAME_STATE_FILE = "game_state.json"
POLL_INTERVAL_SECONDS = 0.1
COOKIE_FILE = "chess_cookies.pkl"

# Unique RGB values for each player's clock from chess.com
CLOCK_COLOR_MAP = {
    'R': "rgb(191, 59, 68)",  # Red
    'B': "rgb(65, 132, 191)", # Blue
    'Y': "rgb(191, 148, 38)", # Yellow
    'G': "rgb(78, 145, 97)",  # Green
}

def load_cookies(driver, cookie_file):
    """Optimized cookie loading without initial navigation."""
    print(f"Loading cookies from '{cookie_file}'...")
    try:
        with open(cookie_file, 'rb') as file:
            cookies = pickle.load(file)
        
        # Enable Network domain to set cookies via CDP
        driver.execute_cdp_cmd('Network.enable', {})
        for cookie in cookies:
            if 'expiry' in cookie:
                del cookie['expiry']
            try:
                driver.execute_cdp_cmd('Network.setCookie', cookie)
            except Exception as e:
                # Ignore specific cookie errors
                pass
        
        print("✅ Cookies loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"⚠️ Cookie file not found. Please log in manually once and save cookies.")
        return False
    except Exception as e:
        print(f"❌ Error loading cookies: {e}")
        return False

def _parse_clock_str(time_str: str) -> float:
    """Converts a 'M:SS' or 'S.s' string to total seconds."""
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3: # H:M:S (rare but possible)
                return float(int(parts[0])*3600 + int(parts[1])*60 + float(parts[2]))
            return float(int(parts[0]) * 60 + float(parts[1]))
        else:
            return float(time_str)
    except (ValueError, IndexError):
        return 0.0

def _fetch_clock_times(driver) -> dict:
    """
    Scrapes the page for all four player clocks.
    """
    clocks_found = {}
    # Regex for time formats: 1:30, 0:45.5, 9.2
    time_format_regex = re.compile(r"^\d{1,2}(:\d{2})?(\.\d)?$")

    try:
        clock_elements = driver.find_elements(By.CSS_SELECTOR, ".clock-component.playerbox-clock")

        for clock_element in clock_elements:
            style = clock_element.get_attribute('style') or ""
            time_text = ""

            # The time is usually inside a span
            spans = clock_element.find_elements(By.TAG_NAME, "span")
            for span in spans:
                txt = span.get_attribute('textContent').strip()
                # Basic validation
                if ":" in txt or "." in txt or txt.isdigit():
                    time_text = txt
                    break

            if time_text:
                seconds = _parse_clock_str(time_text)
                # Determine color based on CSS color style
                for color_char, rgb_val in CLOCK_COLOR_MAP.items():
                    if rgb_val in style:
                        clocks_found[color_char] = seconds
                        break
    except StaleElementReferenceException:
        pass # Retry next poll
    except Exception:
        pass

    return clocks_found

def _convert_raw_coords(match):
    """
    Regex callback to convert 14x14 board coords (d-k, 4-11) 
    to 8x8 engine coords (a-h, 1-8).
    Offset is -3.
    """
    col_char = match.group(1) # e.g. 'f'
    rank_str = match.group(2) # e.g. '5'
    
    # ord('d')=100 -> want 'a'=97. Offset -3.
    new_col_code = ord(col_char) - 3
    new_col_char = chr(new_col_code)
    
    # rank 4 -> want 1. Offset -3.
    new_rank = int(rank_str) - 3
    
    return f"{new_col_char}{new_rank}"

def _standardize_move_notation(raw_title: str) -> str:
    """
    Parses the raw move title from chess.com (e.g. "f5-f6 • 23:40:02")
    and converts it to engine algebraic notation (e.g. "c2-c3").
    """
    if not raw_title: return ""

    # 1. Extract move part (before the bullet point)
    parts = raw_title.split('•')
    move_part = parts[0].strip()
    
    # 2. Handle Game End markers
    # Pass them through; the engine handles 'R' (Resign) and 'T' (Time/Resign).
    if move_part in ('T', 'R', '#'):
        return move_part

    # 3. Cleanup Suffixes
    # Remove checks (+), mates (#), and the specific King Capture marker (#S)
    # Also strip space just in case
    move_part = move_part.replace('#S', '').replace('#', '').replace('+', '').strip()

    # 4. Convert Coordinates
    # Matches any a-n character followed by 1 or 2 digits
    converted_move = re.sub(r'([a-n])(\d+)', _convert_raw_coords, move_part)
    
    return converted_move

def fetch_and_sync_moves(driver):
    """
    Main polling loop.
    """
    last_sent_moves = []
    
    print("\n--- Move Fetcher is running ---")
    print(f"Polling '{args.url}' every {POLL_INTERVAL_SECONDS}s.")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            if "chess.com" not in driver.current_url:
                print("Navigated away from chess.com. Stopping.")
                break

            move_elements = driver.find_elements(By.CSS_SELECTOR, ".moves-pointer")

            current_moves = []
            for move_element in move_elements:
                try:
                    title = move_element.get_attribute('title')
                    if title:
                        std_move = _standardize_move_notation(title)
                        if std_move:
                            current_moves.append(std_move)
                            # --- FIX: REMOVED THE BREAK HERE ---
                            # In FFA, 'R' or 'T' just means that specific player is out.
                            # The game history continues.
                except StaleElementReferenceException:
                    current_moves = None
                    break
            
            if current_moves is None:
                continue

            # Check if state changed
            if current_moves != last_sent_moves:
                detection_time = time.time()
                current_clocks = _fetch_clock_times(driver)
                
                print(f"[{time.strftime('%H:%M:%S')}] New state: {len(current_moves)} moves. Clocks: {current_clocks}")

                payload = {
                    'url': driver.current_url,
                    'moves': current_moves,
                    'clocks': current_clocks,
                    'detection_timestamp': detection_time
                }
                
                # Atomic write
                temp_file = GAME_STATE_FILE + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(payload, f)
                os.replace(temp_file, GAME_STATE_FILE)
                
                last_sent_moves = current_moves

            jitter = random.uniform(-0.02, 0.02)
            time.sleep(max(0.05, POLL_INTERVAL_SECONDS + jitter))

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help="Chess.com game URL")
    # Setup arg kept for compatibility with launcher, but logic is uniform for Chaturaji
    parser.add_argument('--setup', default='modern', help="Ignored for Chaturaji")
    args = parser.parse_args()

    # Selenium Options
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-client-side-phishing-detection")
    options.add_argument("--disable-crash-reporter")
    options.add_argument("--no-crash-upload")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--log-level=3")
    options.add_argument("--silent")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    print("Initializing WebDriver...")
    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        print(f"Error initializing Chrome: {e}")
        sys.exit(1)

    if not load_cookies(driver, COOKIE_FILE):
        print("Warning: Cookies not loaded. You may need to login manually.")

    try:
        print(f"Navigating to {args.url}...")
        driver.get(args.url)
        
        # Wait for board to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".moves-moves-list"))
        )
        print("Game page loaded.")
        
        fetch_and_sync_moves(driver)

    except TimeoutException:
        print("Timed out waiting for page load.")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        driver.quit()