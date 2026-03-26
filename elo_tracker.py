"""
Elo Tracker for Cyberchess-Dojo.

Maintains a running Elo estimate for the AI player (Gemini, GPT-4o, Claude, …)
across games, using the standard FIDE formula with a configurable K-factor.

Results are persisted to ``elo_history.json`` so ratings survive between runs.

Usage::

    from elo_tracker import EloTracker

    elo = EloTracker()                        # load or create history

    delta = elo.update(
        result="0-1",                         # board.result()
        opponent_skill=5,                     # Stockfish skill level
        game_number=1,
    )
    print(f"Elo change: {delta:+.0f}")
    print(elo.report())
"""

import json
import os


# ---------------------------------------------------------------------------
# Stockfish skill → approximate Elo table
# ---------------------------------------------------------------------------

#: Approximate Elo ratings for Stockfish at each skill level (0–20).
#: Based on community benchmarks; intended as a rough guide only.
_STOCKFISH_ELO: dict[int, int] = {
    0:  800,  1: 1000,  2: 1100,  3: 1200,  4: 1350,
    5: 1500,  6: 1650,  7: 1800,  8: 1900,  9: 2000,
    10: 2100, 11: 2200, 12: 2300, 13: 2400, 14: 2500,
    15: 2600, 16: 2700, 17: 2800, 18: 2900, 19: 3000,
    20: 3200,
}

_DEFAULT_STARTING_ELO: float = 1200.0
_K_FACTOR: float = 32.0           # standard K-factor for developing players
_DEFAULT_ELO_FILE: str = "elo_history.json"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def stockfish_elo(skill_level: int) -> int:
    """Return the approximate Elo rating for a given Stockfish skill level (0–20)."""
    clamped = max(0, min(20, skill_level))
    return _STOCKFISH_ELO[clamped]


# ---------------------------------------------------------------------------
# EloTracker
# ---------------------------------------------------------------------------

class EloTracker:
    """
    Tracks the AI player's Elo rating across games.

    State is persisted to a JSON file after every update so it survives
    between separate process runs.

    Attributes:
        current_elo (float): The AI's current estimated Elo rating.
        games_played (int):  Total number of games recorded.
    """

    def __init__(
        self,
        filepath: str = _DEFAULT_ELO_FILE,
        initial_elo: float = _DEFAULT_STARTING_ELO,
    ):
        self.filepath = filepath
        self._data = self._load(initial_elo)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self, initial_elo: float) -> dict:
        """Load existing history from disk, or create fresh state."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath) as f:
                    data = json.load(f)
                    # Validate minimal expected keys
                    if "current_elo" in data and "history" in data:
                        return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"current_elo": initial_elo, "history": []}

    def save(self) -> None:
        """Persist the current Elo state to disk."""
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_elo(self) -> float:
        """The AI's current estimated Elo rating."""
        return float(self._data["current_elo"])

    @property
    def games_played(self) -> int:
        """Total number of completed games recorded."""
        return len(self._data["history"])

    @property
    def history(self) -> list[dict]:
        """Full per-game history list (most recent last)."""
        return list(self._data["history"])

    # ------------------------------------------------------------------
    # Elo calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_score(player_elo: float, opponent_elo: float) -> float:
        """Expected score for *player* vs *opponent* (standard FIDE formula)."""
        return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def update(
        self,
        result: str,
        opponent_skill: int,
        game_number: int,
        ai_color: str = "black",
    ) -> float:
        """
        Update the AI's Elo after a completed game and persist the change.

        Args:
            result:          The game result string from ``board.result()``
                             (``"1-0"``, ``"0-1"``, or ``"1/2-1/2"``).
            opponent_skill:  Stockfish skill level (0–20) used in this game.
            game_number:     Sequential game number (used for record-keeping).
            ai_color:        Colour the AI played (``"black"`` or ``"white"``).

        Returns:
            The Elo change (positive = gain, negative = loss).
        """
        opp_elo = stockfish_elo(opponent_skill)

        # Actual score from the AI's perspective
        if result == "1/2-1/2":
            actual = 0.5
        elif (result == "0-1" and ai_color == "black") or (result == "1-0" and ai_color == "white"):
            actual = 1.0
        else:
            actual = 0.0

        expected = self._expected_score(self.current_elo, opp_elo)
        delta = _K_FACTOR * (actual - expected)
        new_elo = self.current_elo + delta

        entry = {
            "game": game_number,
            "result": result,
            "ai_color": ai_color,
            "opponent_elo": opp_elo,
            "stockfish_skill": opponent_skill,
            "elo_before": round(self.current_elo, 1),
            "elo_after": round(new_elo, 1),
            "delta": round(delta, 1),
        }
        self._data["current_elo"] = new_elo
        self._data["history"].append(entry)
        self.save()
        return delta

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return a human-readable summary of the AI's Elo record."""
        history = self._data["history"]

        if not history:
            return f"No games played yet.  Starting Elo: {self.current_elo:.0f}"

        wins = sum(1 for g in history if g["delta"] > 0)
        draws = sum(1 for g in history if g["delta"] == 0)
        losses = len(history) - wins - draws

        lines = [
            f"── Elo Rating: {self.current_elo:.0f} ──",
            f"   Games: {len(history)}  |  W: {wins}  D: {draws}  L: {losses}",
        ]

        # Show the last 5 games
        shown = history[-5:]
        if len(history) > 5:
            lines.append(f"   (showing last 5 of {len(history)} games)")
        for g in shown:
            sign = "+" if g["delta"] >= 0 else ""
            lines.append(
                f"   Game {g['game']:>3}: {g['result']:<9} | "
                f"vs Stockfish Skill {g['stockfish_skill']} (≈{g['opponent_elo']}) | "
                f"Elo {g['elo_before']:.0f} → {g['elo_after']:.0f} ({sign}{g['delta']:.0f})"
            )
        return "\n".join(lines)
