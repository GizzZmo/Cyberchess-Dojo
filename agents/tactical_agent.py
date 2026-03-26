import chess
from agents.base_agent import BaseChessAgent


class TacticalAgent(BaseChessAgent):
    """
    Specialises in short-range tactics.

    Looks for checks, captures, forks (attacking two pieces at once),
    pins, skewers, and other forcing sequences that win material or
    deliver checkmate.
    """

    name = "Tactical"
    description = "Finds checks, captures, forks, pins, skewers, and forcing sequences"

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        return f"""You are a chess TACTICAL specialist playing as Black against Stockfish.

Your sole focus is forcing moves and material gain:
- Checks that lead to checkmate or win material.
- Captures of undefended or overloaded pieces.
- Forks (one piece attacks two enemy pieces simultaneously).
- Pins and skewers that win material.
- Quiet moves that set up an unavoidable tactical threat.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Scan for all checks and captures in the legal move list.
- Identify the most forcing / highest-value tactical sequence.
- If no immediate tactic exists, choose the move that creates the biggest threat.
- Briefly explain your tactical reasoning (2-3 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
