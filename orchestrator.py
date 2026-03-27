"""
AI Orchestrator for Cyberchess-Dojo.

Coordinates multiple specialised chess agents to produce the best move
for complex positions.  The orchestrator pipeline:

  1. Detect the game phase  (opening / middlegame / endgame).
  2. Select the most relevant agent(s) for that phase.
  3. Consult the selected agents and collect their move + reasoning.
  4. If agents agree, return immediately.
  5. If agents disagree, call the LLM one more time to synthesise a final decision
     that weighs all the expert analyses.
"""

import re
import chess
from collections import Counter
from agents import OpeningAgent, TacticalAgent, PositionalAgent, EndgameAgent


# ---------------------------------------------------------------------------
# Phase-detection thresholds
# ---------------------------------------------------------------------------

# Full-move number at or below which we are still in the opening.
_OPENING_MOVE_LIMIT = 10

# Sum of non-pawn, non-king pieces (both sides) at or below which we are in
# the endgame.  Rough guide: start value ≈ 2*(2+2+2+1) = 14 for minor pieces
# + rooks; a queen counts as 1 in this *piece count* (not material value).
_ENDGAME_PIECE_THRESHOLD = 6

# Default number of independent LLM samples used by best-of-N move selection.
_N_SAMPLES = 3

# Maximum number of characters from agent reasoning included in the ranking prompt.
_MAX_REASONING_LENGTH = 300

# Maximum retry attempts for the LLM-based best-of-N ranking call.
_RANKING_RETRIES = 3


def _piece_count(board: chess.Board) -> int:
    """Return total count of non-pawn, non-king pieces remaining on the board."""
    total = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += len(board.pieces(piece_type, chess.WHITE))
        total += len(board.pieces(piece_type, chess.BLACK))
    return total


def _has_checks_available(board: chess.Board) -> bool:
    """Return True if any legal move gives check — a sign of tactical tension."""
    return any(board.gives_check(move) for move in board.legal_moves)


def _extract_move(text: str) -> str:
    """Extract the last UCI move token from a model response."""
    text = text.replace("`", "").strip()
    matches = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b', text)
    if matches:
        return matches[-1].lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tokens = lines[-1].split()
        return tokens[-1] if tokens else text
    return text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ChessOrchestrator:
    """
    Orchestrates multiple LLM chess agents to select the best move.

    Works with any LLM adapter that implements the ``generate_content`` interface
    (Gemini, OpenAI, Claude, or any custom ``BaseLLMAdapter``).

    Usage::

        orchestrator = ChessOrchestrator(adapter)
        move = orchestrator.get_best_move(board)
    """

    def __init__(self, model):
        self.model = model
        self.opening_agent = OpeningAgent(model)
        self.tactical_agent = TacticalAgent(model)
        self.positional_agent = PositionalAgent(model)
        self.endgame_agent = EndgameAgent(model)

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------

    def _detect_phase(self, board: chess.Board) -> str:
        """Return 'opening', 'middlegame', or 'endgame' for the current position."""
        if board.fullmove_number <= _OPENING_MOVE_LIMIT:
            return "opening"
        if _piece_count(board) <= _ENDGAME_PIECE_THRESHOLD:
            return "endgame"
        return "middlegame"

    # ------------------------------------------------------------------
    # Multi-agent synthesis
    # ------------------------------------------------------------------

    def _synthesise_move(
        self,
        board: chess.Board,
        analyses: list[tuple[str, chess.Move, str]],
    ) -> chess.Move:
        """
        Given conflicting recommendations from multiple agents, ask Gemini to
        act as a grandmaster arbitrator and pick the strongest move.

        ``analyses`` is a list of ``(agent_name, suggested_move, reasoning)`` tuples.
        """
        legal_moves = [m.uci() for m in board.legal_moves]

        advisor_block = "\n\n".join(
            f"--- {name} Agent ---\nSuggested move: {move.uci()}\nReasoning:\n{reasoning}"
            for name, move, reasoning in analyses
        )

        prompt = f"""You are a chess grandmaster arbitrating between specialist AI advisors.
Each advisor has analysed the current position and suggests a move for Black.

Current board (FEN): {board.fen()}
Legal moves: {', '.join(legal_moves)}

{advisor_block}

Your task:
1. Critically evaluate each advisor's suggestion and reasoning.
2. Determine which move is objectively strongest for Black in this position.
3. Briefly justify your final decision (2-3 sentences).
4. On the very last line, write ONLY your chosen UCI move (e.g. e7e5).
"""

        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                raw = response.text.strip()
                move_str = _extract_move(raw)
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    print(f"  [Orchestrator] Synthesis chose: {move.uci()}")
                    return move
                prompt += f"\n\nERROR: '{move_str}' is not legal. Choose from: {', '.join(legal_moves)}"
            except Exception as e:
                print(f"  [Orchestrator] Synthesis error on attempt {attempt + 1}: {e}")

        # If synthesis itself fails, defer to the primary agent's recommendation.
        fallback = analyses[0][1]
        print(f"  [Orchestrator] Synthesis failed — deferring to {analyses[0][0]} agent: {fallback.uci()}")
        return fallback

    # ------------------------------------------------------------------
    # Best-of-N ranking
    # ------------------------------------------------------------------

    def _rank_candidates(
        self,
        board: chess.Board,
        candidates: list[tuple[chess.Move, str]],
    ) -> chess.Move:
        """
        Select the single strongest move from a list of ``(move, reasoning)``
        candidate samples using an LLM-based ranking call.

        If all samples agree (or only one candidate is provided) the move is
        returned immediately without an extra model call.  If the ranking call
        fails, the most-frequently-suggested candidate is used as a fallback.
        """
        if not candidates:
            raise ValueError("No candidates to rank.")

        unique_moves = {m for m, _ in candidates}
        if len(unique_moves) == 1:
            move = candidates[0][0]
            print(f"  [Orchestrator] All {len(candidates)} samples agree on {move.uci()} — skipping ranking.")
            return move

        legal_moves = [m.uci() for m in board.legal_moves]

        # Build one entry per unique move (avoids duplicate blocks in the prompt).
        seen: set[chess.Move] = set()
        unique_candidates: list[tuple[chess.Move, str]] = []
        for move, reasoning in candidates:
            if move not in seen:
                seen.add(move)
                unique_candidates.append((move, reasoning))

        candidate_block = "\n\n".join(
            f"Candidate {i + 1}: {move.uci()}\nReasoning: {reasoning[:_MAX_REASONING_LENGTH]}"
            for i, (move, reasoning) in enumerate(unique_candidates)
        )

        prompt = f"""You are a chess grandmaster evaluating candidate moves for Black.

Position (FEN): {board.fen()}
Legal moves: {', '.join(legal_moves)}

The following moves were proposed by analysis agents via best-of-N sampling:

{candidate_block}

Task:
1. Evaluate each candidate considering material, king safety, piece activity, and long-term prospects.
2. Select the single strongest move for Black.
3. Briefly justify your choice (2-3 sentences).
4. On the very last line, write ONLY the chosen UCI move (e.g. e7e5).
"""

        for attempt in range(_RANKING_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                raw = response.text.strip()
                move_str = _extract_move(raw)
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    print(f"  [Orchestrator] Best-of-N ranking selected: {move.uci()}")
                    return move
                prompt += f"\n\nERROR: '{move_str}' is not legal. Choose from: {', '.join(legal_moves)}"
            except Exception as e:
                print(f"  [Orchestrator] Ranking error on attempt {attempt + 1}: {e}")

        # Fallback: most frequently suggested candidate.
        best_move = Counter(move for move, _ in candidates).most_common(1)[0][0]
        print(f"  [Orchestrator] Ranking failed — using most frequent candidate: {best_move.uci()}")
        return best_move

    # ------------------------------------------------------------------
    # Main entry point (best-of-N)
    # ------------------------------------------------------------------

    def get_best_move(self, board: chess.Board, n: int = _N_SAMPLES) -> chess.Move:
        """
        Select the best move for Black using best-of-N sampling.

        *n* candidate moves are independently sampled from the phase-appropriate
        agent(s) and a ranking call selects the strongest option.

        Phase routing mirrors :meth:`get_move`:

        * **Opening** (moves 1-10): *n* samples from OpeningAgent.
        * **Endgame** (≤ 6 non-pawn pieces): *n* samples from EndgameAgent.
        * **Middlegame with tactical tension**: ⌈n/2⌉ from TacticalAgent +
          ⌊n/2⌋ from PositionalAgent.
        * **Quiet middlegame**: ⌈n/2⌉ from PositionalAgent + ⌊n/2⌋ from
          TacticalAgent.
        """
        phase = self._detect_phase(board)
        print(f"  [Orchestrator] Phase: {phase} | Move: {board.fullmove_number} | N={n}")

        candidates: list[tuple[chess.Move, str]] = []

        if phase == "opening":
            print(f"  [Orchestrator] → OpeningAgent ×{n}")
            candidates = self.opening_agent.get_move_candidates(board, n)

        elif phase == "endgame":
            print(f"  [Orchestrator] → EndgameAgent ×{n}")
            candidates = self.endgame_agent.get_move_candidates(board, n)

        else:
            n_primary = (n + 1) // 2    # ceiling division
            n_secondary = n // 2        # floor division
            if _has_checks_available(board):
                print(
                    f"  [Orchestrator] Tactical tension → "
                    f"TacticalAgent ×{n_primary} + PositionalAgent ×{n_secondary}"
                )
                candidates += self.tactical_agent.get_move_candidates(board, n_primary)
                candidates += self.positional_agent.get_move_candidates(board, n_secondary)
            else:
                print(
                    f"  [Orchestrator] Quiet middlegame → "
                    f"PositionalAgent ×{n_primary} + TacticalAgent ×{n_secondary}"
                )
                candidates += self.positional_agent.get_move_candidates(board, n_primary)
                candidates += self.tactical_agent.get_move_candidates(board, n_secondary)

        return self._rank_candidates(board, candidates)

    # ------------------------------------------------------------------
    # Original single-sample entry point (kept for backward compatibility)
    # ------------------------------------------------------------------

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Select the best move for Black in the given position.

        Routing logic
        -------------
        * **Opening** (moves 1-10): OpeningAgent only.
        * **Endgame**  (≤ 6 non-pawn pieces remain): EndgameAgent only.
        * **Middlegame with checks available** (tactical tension):
          TacticalAgent (primary) + PositionalAgent (secondary) → synthesise if they disagree.
        * **Quiet middlegame**: PositionalAgent (primary) + TacticalAgent (secondary)
          → synthesise if they disagree.
        """
        phase = self._detect_phase(board)
        print(f"  [Orchestrator] Phase: {phase} | Move: {board.fullmove_number}")

        # ---- Single-agent phases ----------------------------------------
        if phase == "opening":
            print("  [Orchestrator] → OpeningAgent")
            move, _ = self.opening_agent.get_move(board)
            return move

        if phase == "endgame":
            print("  [Orchestrator] → EndgameAgent")
            move, _ = self.endgame_agent.get_move(board)
            return move

        # ---- Middlegame: two agents + optional synthesis -----------------
        if _has_checks_available(board):
            print("  [Orchestrator] Tactical tension detected → TacticalAgent + PositionalAgent")
            primary_agent, secondary_agent = self.tactical_agent, self.positional_agent
            primary_name, secondary_name = "Tactical", "Positional"
        else:
            print("  [Orchestrator] Quiet middlegame → PositionalAgent + TacticalAgent")
            primary_agent, secondary_agent = self.positional_agent, self.tactical_agent
            primary_name, secondary_name = "Positional", "Tactical"

        primary_move, primary_reasoning = primary_agent.get_move(board)
        secondary_move, secondary_reasoning = secondary_agent.get_move(board)

        analyses = [
            (primary_name, primary_move, primary_reasoning),
            (secondary_name, secondary_move, secondary_reasoning),
        ]

        if primary_move == secondary_move:
            print(f"  [Orchestrator] Both agents agree on {primary_move.uci()} — no synthesis needed")
            return primary_move

        # Agents disagree: let Gemini synthesise a final decision.
        print(
            f"  [Orchestrator] Agents disagree "
            f"({primary_name}: {primary_move.uci()} vs {secondary_name}: {secondary_move.uci()}) "
            "— synthesising…"
        )
        return self._synthesise_move(board, analyses)


# ---------------------------------------------------------------------------
# Convenience fallback (drop-in replacement for the original get_gemini_move)
# ---------------------------------------------------------------------------

def get_orchestrated_move(board: chess.Board, model) -> chess.Move:
    """Thin wrapper that creates a fresh orchestrator and returns its chosen move (best-of-N)."""
    return ChessOrchestrator(model).get_best_move(board)


if __name__ == "__main__":
    # Quick smoke-test: verify the orchestrator can be instantiated and phase
    # detection works on the starting position (no API key required).
    import os as _os
    _board = chess.Board()
    _api_key = _os.environ.get("GOOGLE_API_KEY", "")
    if not _api_key or _api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("Set GOOGLE_API_KEY to run the full orchestrator smoke-test.")
        print(f"Piece count on starting position: {_piece_count(_board)}  (expected: 14)")
        print(f"Checks available from start: {_has_checks_available(_board)}  (expected: False)")
    else:
        import google.generativeai as _genai
        _genai.configure(api_key=_api_key)
        _model = _genai.GenerativeModel("gemini-1.5-flash")
        _orch = ChessOrchestrator(_model)
        print(f"Phase: {_orch._detect_phase(_board)!r}  (expected: 'opening')")
        print("ChessOrchestrator instantiated successfully.")
