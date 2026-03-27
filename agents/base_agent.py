"""
Base chess agent for Cyberchess-Dojo.

All specialist agents (Opening, Tactical, Positional, Endgame) inherit from
``BaseChessAgent``, which provides:
- Shared retry logic for illegal / malformed LLM moves.
- UCI move extraction via regex with a line-fallback heuristic.
- Random-move fallback when all retries are exhausted.
- ``get_move_candidates`` for best-of-N sampling.

Subclasses must implement ``_build_prompt`` to return a domain-specific
system / user prompt.
"""

import re
import random
import chess


class BaseChessAgent:
    """
    Abstract base class for all LLM-powered chess agents.

    Subclasses must implement ``_build_prompt``, which returns the specialised
    system/user prompt for that agent's area of expertise.
    """

    name = "Base"
    description = "Generic chess agent"

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        """Return the specialised prompt for this agent. Must be overridden."""
        raise NotImplementedError("Subclasses must implement _build_prompt")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_move(text: str) -> str:
        """
        Extract the first/last UCI move token from the agent's response text.
        UCI format: <file><rank><file><rank>[promotion]  e.g. e2e4, e7e8q
        Returns the last match found (agents are asked to put the move last).
        """
        text = text.replace("`", "").strip()
        # Regex: letter a-h, digit 1-8, letter a-h, digit 1-8, optional promotion
        matches = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b', text)
        if matches:
            return matches[-1].lower()
        # Fallback: last non-empty token on the last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            tokens = lines[-1].split()
            return tokens[-1] if tokens else text
        return text

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def get_move_candidates(self, board: chess.Board, n: int = 3) -> list[tuple[chess.Move, str]]:
        """
        Generate *n* candidate ``(move, reasoning)`` pairs via best-of-N sampling.

        Each call to the model is an independent sample.  Duplicate moves are
        retained so that sample frequency can act as a confidence signal during
        downstream ranking.
        """
        return [self.get_move(board) for _ in range(n)]

    def get_move(self, board: chess.Board, retries: int = 3) -> tuple[chess.Move, str]:
        """
        Ask this agent for the best move on the given board.

        Returns a ``(move, reasoning)`` tuple where *reasoning* is the raw
        text response from the model.  Falls back to a random legal move if
        every retry fails.
        """
        legal_moves = [m.uci() for m in board.legal_moves]
        prompt = self._build_prompt(board, legal_moves)

        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                raw = response.text.strip()
                move_str = self._extract_move(raw)
                move = chess.Move.from_uci(move_str)

                if move in board.legal_moves:
                    return move, raw

                print(f"  [{self.name}] Illegal move '{move_str}' (attempt {attempt + 1}). Retrying...")
                prompt += (
                    f"\n\nERROR: '{move_str}' is NOT a legal move. "
                    f"You MUST choose from: {', '.join(legal_moves)}"
                )

            except Exception as e:
                print(f"  [{self.name}] Parse error on attempt {attempt + 1}: {e}")
                prompt += "\n\nERROR: Invalid format. End your response with the UCI move on its own line (e.g. e7e5)."

        fallback = random.choice(list(board.legal_moves))
        print(f"  [{self.name}] All retries exhausted — playing random move: {fallback.uci()}")
        return fallback, f"fallback: random move ({fallback.uci()})"
