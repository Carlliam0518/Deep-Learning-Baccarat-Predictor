# live_capture.py
import json
import time
import cv2
import mss
import numpy as np
from PIL import Image
import torch

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    CLASS_NAMES_PATH,
    CARD_ROIS,
    CONFIDENCE_THRESHOLD,
    SCAN_INTERVAL_SEC,
)
from utils import banner, ok, info, warn, show_table
from data_loader import get_transforms
from model_builder import build_model
from baccarat_engine import BaccaratShoeEngine, HandStateTracker

# ---------------------------------------------------------------- 1. Baccarat Card Point Mapper
BACCARAT_WORD_MAP = {
    "ace": 1,
    "two": 2, "2": 2,
    "three": 3, "3": 3,
    "four": 4, "4": 4,
    "five": 5, "5": 5,
    "six": 6, "6": 6,
    "seven": 7, "7": 7,
    "eight": 8, "8": 8,
    "nine": 9, "9": 9,
    "ten": 0, "10": 0,
    "jack": 0,
    "queen": 0,
    "king": 0,
    "joker": 0,
}

def card_label_to_baccarat_value(class_label):
    """
    Converts any predicted playing card class name (e.g., 'eight of diamonds'
    or 'king of spades') into its Baccarat point value (0 to 9).
    """
    tokens = class_label.lower().replace("-", " ").replace("_", " ").split()
    for token in tokens:
        if token in BACCARAT_WORD_MAP:
            return BACCARAT_WORD_MAP[token]
    return 0

# ---------------------------------------------------------------- 2. Model & Checkpoint Loader Helper
def load_trained_detector(model_name="MobileNetV3"):
    """
    Loads saved class names and restores the trained .pth checkpoint
    for fast real-time screen inference.
    """
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLASS_NAMES_PATH}. Run 'python main.py' first to train and save checkpoints."
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {ckpt_path}. Run 'python main.py' first."
        )

    model = build_model(model_name, num_classes=len(class_names), freeze_backbone=False)
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    eval_transform = get_transforms()["eval"]
    ok(f"Loaded {model_name} detector from {ckpt_path} ({len(class_names)} card classes)")
    return model, class_names, eval_transform

# ---------------------------------------------------------------- 3. Single Bounding-Box Capture & Prediction Helper
def predict_screen_region(sct, bbox, model, class_names, eval_transform):
    """
    Grabs a specific screen bounding box (top, left, width, height),
    converts pixels to a PIL Image, runs model inference, and returns
    the card name, confidence score, Baccarat value, and raw RGB frame.
    """
    raw_grab = np.array(sct.grab(bbox))
    rgb_frame = cv2.cvtColor(raw_grab, cv2.COLOR_BGRA2RGB)
    pil_img = Image.fromarray(rgb_frame)

    input_tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    confidence = float(conf.item())
    card_name = class_names[int(pred_idx.item())]
    baccarat_val = card_label_to_baccarat_value(card_name)

    return {
        "card_name": card_name,
        "confidence": round(confidence, 4),
        "baccarat_val": baccarat_val,
        "rgb_frame": rgb_frame,
    }

# ---------------------------------------------------------------- 4. Full Baccarat Table Scanner Helper
def scan_baccarat_table(sct, model, class_names, eval_transform, rois=CARD_ROIS, conf_thresh=CONFIDENCE_THRESHOLD):
    """
    Scans all defined Player and Banker card bounding boxes on screen,
    filters out low-confidence empty slots, and computes Baccarat hand totals.
    """
    slot_results = {}
    player_points = 0
    banker_points = 0

    for slot_name, bbox in rois.items():
        res = predict_screen_region(sct, bbox, model, class_names, eval_transform)
        is_card_present = res["confidence"] >= conf_thresh

        if is_card_present:
            if slot_name.startswith("Player"):
                player_points = (player_points + res["baccarat_val"]) % 10
            elif slot_name.startswith("Banker"):
                banker_points = (banker_points + res["baccarat_val"]) % 10

        slot_results[slot_name] = {
            "Detected Card": res["card_name"] if is_card_present else "Empty / Waiting",
            "Confidence": res["confidence"],
            "Baccarat Pts": res["baccarat_val"] if is_card_present else "-",
        }

    return slot_results, player_points, banker_points

# ---------------------------------------------------------------- 5. Live Capture + Probability Engine Coordinator
def start_live_capture(model_name="MobileNetV3", num_decks=8, simulations=10000, max_scans=None):
    """
    Runs continuous live screen capture, deduplicates cards across frames,
    updates the 8-deck shoe inventory, and outputs real-time probabilities.
    """
    banner("LIVE CAPTURE & ENGINE | Real-Time Baccarat Vision + Shoe Tracker")
    model, class_names, eval_transform = load_trained_detector(model_name)

    shoe_engine = BaccaratShoeEngine(num_decks=num_decks, simulations=simulations)
    hand_tracker = HandStateTracker(empty_reset_scans=2)

    # Compute initial baseline shoe probabilities
    initial_probs = shoe_engine.calculate_probabilities()
    show_table(initial_probs, col_name="Initial 8-Deck Baseline", index_name="Metric")

    info(f"Monitoring {len(CARD_ROIS)} card slots every {SCAN_INTERVAL_SEC}s (Press Ctrl+C to stop)...")

    scan_count = 0
    with mss.mss() as sct:
        try:
            while True:
                scan_count += 1
                slot_results, p_total, b_total = scan_baccarat_table(
                    sct, model, class_names, eval_transform
                )

                # Filter out cards already counted in earlier frames of the same hand
                newly_dealt = hand_tracker.extract_newly_dealt_cards(slot_results)

                if newly_dealt:
                    print(f"\n[SCAN #{scan_count}] NEW CARD(S) DETECTED | PLAYER: {p_total} | BANKER: {b_total}")
                    for card in newly_dealt:
                        shoe_engine.remove_card(
                            baccarat_val=card["baccarat_val"],
                            slot_name=card["slot"],
                            card_name=card["card_name"],
                        )

                    # Recalculate probabilities whenever the shoe changes
                    live_probs = shoe_engine.calculate_probabilities()
                    print(
                        f"  -> NEXT HAND ODDS | "
                        f"Player: {live_probs['Player Win (%)']}% (EV: {live_probs['Player EV (%)']:+.3f}%) | "
                        f"Banker: {live_probs['Banker Win (%)']}% (EV: {live_probs['Banker EV (%)']:+.3f}%) | "
                        f"Tie: {live_probs['Tie (%)']}%"
                    )
                    print(f"  -> OPTIMAL CALL   | {live_probs['Recommended Edge']} | Cards Left: {live_probs['Cards Remaining']}")

                if max_scans is not None and scan_count >= max_scans:
                    ok(f"Completed {max_scans} scheduled scans.")
                    break

                time.sleep(SCAN_INTERVAL_SEC)

        except KeyboardInterrupt:
            warn("Live capture and probability engine stopped by user.")

if __name__ == "__main__":
    start_live_capture(model_name="MobileNetV3", num_decks=8, simulations=10000)