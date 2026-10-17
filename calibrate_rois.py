# calibrate_rois.py
import json
import time
import cv2
import mss
import numpy as np

from config import CHECKPOINT_DIR, ROIS_JSON_PATH, DEFAULT_CARD_ROIS
from utils import banner, ok, info, warn, show_table

SLOT_NAMES = list(DEFAULT_CARD_ROIS.keys())

# ---------------------------------------------------------------- 1. Full Screen Snapshot Helper
def capture_full_screen(monitor_index=1, delay_sec=3):
    """
    Waits a few seconds so you can switch to your Baccarat browser window,
    then captures a full-screen BGR image for OpenCV bounding-box selection.
    """
    info(f"Switch to your Baccarat website window now! Taking screenshot in {delay_sec} seconds...")
    for remaining in range(delay_sec, 0, -1):
        print(f"    Capturing in {remaining}...")
        time.sleep(1)

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        raw_shot = np.array(sct.grab(monitor))
        bgr_frame = cv2.cvtColor(raw_shot, cv2.COLOR_BGRA2BGR)

    ok(f"Captured screen resolution: {bgr_frame.shape[1]}x{bgr_frame.shape[0]}")
    return bgr_frame

# ---------------------------------------------------------------- 2. Visual Overlay Helper
def draw_existing_rois(base_image, calibrated_rois, current_slot):
    """
    Draws all previously selected card boxes in green (Player) or orange (Banker)
    and displays instructions at the top of the calibration window.
    """
    canvas = base_image.copy()

    # Draw top instruction bar
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (25, 25, 25), -1)
    instruction = (
        f"SELECTING: [{current_slot}]  |  "
        "Click & Drag Box -> Press ENTER or SPACE to confirm  |  Press C to cancel"
    )
    cv2.putText(
        canvas, instruction, (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
    )

    # Draw already calibrated boxes so you don't overlap them
    for slot, bbox in calibrated_rois.items():
        x, y, w, h = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
        color = (255, 180, 50) if slot.startswith("Player") else (50, 140, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas, slot, (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )

    return canvas

# ---------------------------------------------------------------- 3. Single Slot ROI Selector Helper
def select_single_roi(base_image, calibrated_rois, slot_name):
    """
    Opens an interactive OpenCV window for one card slot and returns
    the bounding box formatted for mss: {'top', 'left', 'width', 'height'}.
    """
    win_name = "Baccarat ROI Calibrator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    canvas = draw_existing_rois(base_image, calibrated_rois, slot_name)
    x, y, w, h = cv2.selectROI(win_name, canvas, showCrosshair=True, fromCenter=False)

    if w == 0 or h == 0:
        warn(f"No box drawn for {slot_name}. Using default fallback coordinates.")
        return DEFAULT_CARD_ROIS[slot_name]

    roi_dict = {
        "top": int(y),
        "left": int(x),
        "width": int(w),
        "height": int(h),
    }
    ok(f"{slot_name} locked -> {roi_dict}")
    return roi_dict

# ---------------------------------------------------------------- 4. Master Calibration Workflow
def run_calibration(slot_names=SLOT_NAMES, delay_sec=3):
    """
    Captures the screen, walks the user through dragging boxes for all 6
    Baccarat card slots, saves to checkpoints/card_rois.json, and prints
    the ready-to-paste Python dictionary.
    """
    banner("CALIBRATION TOOL | Interactive Screen Bounding-Box Selector")
    screen_bgr = capture_full_screen(delay_sec=delay_sec)

    calibrated_rois = {}
    for slot_name in slot_names:
        info(f"Draw box for: {slot_name} (Press ENTER/SPACE when done)")
        calibrated_rois[slot_name] = select_single_roi(
            screen_bgr, calibrated_rois, slot_name
        )

    cv2.destroyAllWindows()

    # 1. Save to JSON so config.py and live_capture.py load it automatically
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROIS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(calibrated_rois, f, indent=4)
    ok(f"Saved calibrated coordinates to {ROIS_JSON_PATH} (auto-loaded by config.py!)")

    # 2. Display summary table
    show_table(calibrated_rois, col_name="Coordinates (top, left, width, height)", index_name="Card Slot")

    # 3. Also print ready-to-paste Python code for reference
    print("\n# Ready-to-paste CARD_ROIS dictionary (if you ever want to hardcode it):")
    print("CARD_ROIS = " + json.dumps(calibrated_rois, indent=4))
    return calibrated_rois

if __name__ == "__main__":
    run_calibration()