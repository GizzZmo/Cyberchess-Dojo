"""
Adaptive training controls for Cyberchess-Dojo.

This module provides a lightweight automated curriculum that adjusts game
configuration over time to help the Black AI improve Elo progression.
"""

import json
import os


_DEFAULT_PROGRESS_FILE = "adaptive_progress.json"
_SMOOTHING_CURRENT_WEIGHT = 0.7
_SMOOTHING_PREVIOUS_WEIGHT = 0.3
_RECOVERY_THRESHOLD = 0.30
_RECOVERY_HYSTERESIS_THRESHOLD = 0.33
_CHALLENGE_THRESHOLD = 0.70
_CHALLENGE_HYSTERESIS_THRESHOLD = 0.67


def _score_from_ai_perspective(result: str, ai_color: str = "black") -> float:
    """Convert a PGN result string into a score from the AI's perspective."""
    if result == "1/2-1/2":
        return 0.5
    if (result == "0-1" and ai_color == "black") or (result == "1-0" and ai_color == "white"):
        return 1.0
    return 0.0


class AdaptiveTrainingManager:
    """
    Adjust Stockfish challenge and sampling strength from recent performance.

    Strategy:
      - If recent score is low, make the teacher slightly easier and increase
        best-of-N to improve move quality.
      - If recent score is high, increase challenge to keep learning pressure.
      - Otherwise keep a stable regime.
    """

    def __init__(self, filepath: str = _DEFAULT_PROGRESS_FILE):
        self.filepath = filepath
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"history": []}

    def save(self) -> None:
        """Persist adaptation history."""
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _recent_score(history: list[dict], span: int = 8) -> float:
        """Return the average AI score over the last *span* games."""
        if not history:
            return 0.5
        recent = history[-span:]
        scores = [_score_from_ai_perspective(g.get("result", "*"), g.get("ai_color", "black")) for g in recent]
        return sum(scores) / len(scores)

    def plan_next_game(
        self,
        *,
        base_skill: int,
        base_time: float,
        base_best_of_n: int,
        elo_history: list[dict],
    ) -> dict:
        """
        Produce tuned parameters for the next game.

        Returns a dict with:
          - stockfish_skill
          - stockfish_time
          - best_of_n
          - recent_score
          - regime
        """
        recent_raw = self._recent_score(elo_history)
        previous_plan = self._data.get("history", [])[-1] if self._data.get("history") else {}
        previous_recent = float(previous_plan.get("recent_score", recent_raw))
        # Exponential smoothing to reduce oscillation in regime switching.
        recent = (
            _SMOOTHING_CURRENT_WEIGHT * recent_raw
            + _SMOOTHING_PREVIOUS_WEIGHT * previous_recent
        )

        skill = int(base_skill)
        stockfish_time = float(base_time)
        best_of_n = int(base_best_of_n)
        regime = "stable"
        previous_regime = str(previous_plan.get("regime", "stable"))

        # Hysteresis thresholds: avoid rapid flips around boundaries.
        low_threshold = (
            _RECOVERY_HYSTERESIS_THRESHOLD
            if previous_regime == "recovery"
            else _RECOVERY_THRESHOLD
        )
        high_threshold = (
            _CHALLENGE_HYSTERESIS_THRESHOLD
            if previous_regime == "challenge"
            else _CHALLENGE_THRESHOLD
        )

        if recent < low_threshold:
            regime = "recovery"
            skill = max(0, skill - 1)
            stockfish_time = max(0.01, stockfish_time * 0.8)
            best_of_n = min(10, best_of_n + 1)
        elif recent > high_threshold:
            regime = "challenge"
            skill = min(20, skill + 1)
            stockfish_time = min(60.0, stockfish_time * 1.2)
            best_of_n = min(10, best_of_n + 1)

        plan = {
            "stockfish_skill": skill,
            "stockfish_time": round(stockfish_time, 3),
            "best_of_n": best_of_n,
            "recent_score_raw": round(recent_raw, 3),
            "recent_score": round(recent, 3),
            "regime": regime,
        }
        self._data["history"].append(plan)
        self.save()
        return plan
