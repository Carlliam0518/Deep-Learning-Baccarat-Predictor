# baccarat_engine.py
import json
import numpy as np
import pandas as pd
from IPython.display import display

from config import CHECKPOINT_DIR, SEED
from utils import ok, info, warn

LIVE_STATE_JSON_PATH = CHECKPOINT_DIR / "live_baccarat_state.json"

class BaccaratShoeEngine:
    """
    Manages an 8-deck (416-card) Baccarat shoe, removes cards detected by
    live_capture.py, runs fast vectorized/Monte Carlo simulations using exact
    Punto Banco third-card rules, and computes live Expected Value (EV).
    """
    def __init__(self, num_decks=8, simulations=10000):
        self.num_decks = num_decks
        self.simulations = simulations
        self.rng = np.random.default_rng(SEED)
        self.reset_shoe()

    # ---------------------------------------------------------------- 1. Shoe State Helpers
    def reset_shoe(self):
        """Initializes a fresh 8-deck shoe sorted by Baccarat point values (0-9)."""
        # Per deck: sixteen 0-value cards (10, J, Q, K) and four of each 1-9
        self.counts = {0: 16 * self.num_decks}
        for val in range(1, 10):
            self.counts[val] = 4 * self.num_decks

        self.dealt_history = []
        self.hands_completed = 0
        ok(f"Initialized fresh {self.num_decks}-deck shoe ({self.total_remaining()} cards)")

    def total_remaining(self):
        """Returns the total number of cards left in the shoe."""
        return sum(self.counts.values())

    def remove_card(self, baccarat_val, slot_name="Unknown", card_name="Unknown"):
        """Deducts a single detected card from the remaining shoe inventory."""
        val = int(baccarat_val)
        if self.counts.get(val, 0) > 0:
            self.counts[val] -= 1
            self.dealt_history.append({
                "slot": slot_name,
                "card_name": card_name,
                "baccarat_val": val,
                "remaining_in_shoe": self.total_remaining(),
            })
            info(
                f"Shoe Updated -> Removed [{card_name} (Pts: {val})] from {slot_name} | "
                f"Cards left: {self.total_remaining()}"
            )
            return True
        else:
            warn(f"Card value {val} depleted in shoe! Resetting shoe automatically.")
            self.reset_shoe()
            return False

    # ---------------------------------------------------------------- 2. Official Third-Card Rules Helper
    @staticmethod
    def banker_draws_third(banker_total, player_third_card):
        """
        Applies the official Casino Punto Banco Banker third-card drawing tableau
        when the Player has drawn a third card.
        """
        if banker_total <= 2:
            return True
        if banker_total == 3:
            return player_third_card != 8
        if banker_total == 4:
            return player_third_card in (2, 3, 4, 5, 6, 7)
        if banker_total == 5:
            return player_third_card in (4, 5, 6, 7)
        if banker_total == 6:
            return player_third_card in (6, 7)
        return False

    def simulate_single_hand(self, deck_array):
        """
        Draws up to 6 cards without replacement from the current shoe array
        and resolves the winner ('Player', 'Banker', or 'Tie').
        """
        draw = self.rng.choice(deck_array, size=6, replace=False)
        p1, b1, p2, b2, c5, c6 = int(draw[0]), int(draw[1]), int(draw[2]), int(draw[3]), int(draw[4]), int(draw[5])

        p_total = (p1 + p2) % 10
        b_total = (b1 + b2) % 10

        # Check Natural 8 or 9
        if p_total < 8 and b_total < 8:
            if p_total <= 5:
                # Player draws 3rd card (c5)
                p_third = c5
                p_total = (p_total + p_third) % 10
                if self.banker_draws_third(b_total, p_third):
                    b_total = (b_total + c6) % 10
            else:
                # Player stands on 6 or 7; Banker draws on 0-5
                if b_total <= 5:
                    b_total = (b_total + c5) % 10

        if p_total > b_total:
            return "Player"
        elif b_total > p_total:
            return "Banker"
        return "Tie"

    # ---------------------------------------------------------------- 3. Probability & EV Calculator
    def calculate_probabilities(self):
        """
        Runs Monte Carlo simulations over the exact remaining shoe composition
        and returns win probabilities, Expected Value (EV), and the best bet.
        """
        if self.total_remaining() < 14:
            warn("Fewer than 14 cards remaining (Cut card reached). Resetting shoe.")
            self.reset_shoe()

        # Expand remaining counts into a flat array of card values for fast sampling
        deck_array = np.concatenate([
            np.full(count, val, dtype=np.int8)
            for val, count in self.counts.items() if count > 0
        ])

        wins = {"Player": 0, "Banker": 0, "Tie": 0}
        for _ in range(self.simulations):
            outcome = self.simulate_single_hand(deck_array)
            wins[outcome] += 1

        p_win = wins["Player"] / self.simulations
        b_win = wins["Banker"] / self.simulations
        t_win = wins["Tie"] / self.simulations

        # Standard Payouts: Player (1:1), Banker (0.95:1 with 5% commission), Tie (8:1)
        ev_player = (p_win * 1.0) - (b_win * 1.0)
        ev_banker = (b_win * 0.95) - (p_win * 1.0)
        ev_tie = (t_win * 8.0) - ((p_win + b_win) * 1.0)

        evs = {"Player": ev_player, "Banker": ev_banker, "Tie": ev_tie}
        best_bet = max(evs, key=evs.get)

        report = {
            "Cards Remaining": self.total_remaining(),
            "Player Win (%)": round(p_win * 100, 2),
            "Banker Win (%)": round(b_win * 100, 2),
            "Tie (%)": round(t_win * 100, 2),
            "Player EV (%)": round(ev_player * 100, 3),
            "Banker EV (%)": round(ev_banker * 100, 3),
            "Tie EV (%)": round(ev_tie * 100, 3),
            "Recommended Edge": f"{best_bet.upper()} ({evs[best_bet]*100:+.3f}% EV)",
        }

        self.export_live_state(report)
        return report

    # ---------------------------------------------------------------- 4. JSON Dashboard Export Helper
    def export_live_state(self, latest_report):
        """
        Exports the current shoe inventory, probability metrics, and recent cards
        to checkpoints/live_baccarat_state.json so any local HTML UI can read it.
        """
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "shoe_counts": {str(k): v for k, v in self.counts.items()},
            "probabilities": latest_report,
            "recent_cards": self.dealt_history[-12:],
        }
        with open(LIVE_STATE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)


# ---------------------------------------------------------------- 5. Duplicate-Frame State Filter
class HandStateTracker:
    """
    Prevents counting the same physical card multiple times across consecutive
    1-second screen scans. Detects when a new card appears in a slot and when
    the dealer clears the table for the next hand.
    """
    def __init__(self, empty_reset_scans=2):
        self.active_slots = {}
        self.empty_streak = 0
        self.empty_reset_scans = empty_reset_scans

    def extract_newly_dealt_cards(self, slot_results):
        """
        Compares current screen scan against active table memory.
        Returns a list of ONLY newly dealt cards in this frame.
        """
        new_cards = []
        any_card_visible = False

        for slot_name, info_dict in slot_results.items():
            card_name = info_dict["Detected Card"]
            if card_name != "Empty / Waiting":
                any_card_visible = True
                # If this slot was previously empty, lock in the new card!
                if slot_name not in self.active_slots:
                    self.active_slots[slot_name] = card_name
                    new_cards.append({
                        "slot": slot_name,
                        "card_name": card_name,
                        "baccarat_val": int(info_dict["Baccarat Pts"]),
                    })

        if not any_card_visible:
            self.empty_streak += 1
            if self.empty_streak >= self.empty_reset_scans and self.active_slots:
                ok("Table cleared by dealer -> Ready for next hand!")
                self.active_slots.clear()
        else:
            self.empty_streak = 0

        return new_cards