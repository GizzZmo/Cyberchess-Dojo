"""
AI Orchestrator for Cyberchess-Dojo.

Coordinates multiple specialised Gemini chess agents to produce the best move
for complex positions.  The orchestrator pipeline:

  1. Detect the game phase  (opening / middlegame / endgame).
  2. Select the most relevant agent(s) for that phase.
  3. Consult the selected agents and collect their move + reasoning.
  4. If agents agree, return immediately.
  5. If agents disagree, call Gemini one more time to synthesise a final decision
     that weighs all the expert analyses.
"""

import re
import chess
import google.generativeai as genai
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
    Orchestrates multiple Gemini chess agents to select the best move.

    Usage::

        orchestrator = ChessOrchestrator(model)
        move = orchestrator.get_move(board)
    """

    def __init__(self, model: genai.GenerativeModel):
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
    # Main entry point
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

def get_orchestrated_move(board: chess.Board, model: genai.GenerativeModel) -> chess.Move:
    """Thin wrapper that creates a fresh orchestrator and returns its chosen move."""
    return ChessOrchestrator(model).get_move(board)


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
