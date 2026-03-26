import chess
from agents.base_agent import BaseChessAgent


class OpeningAgent(BaseChessAgent):
    """
    Specialises in opening play.

    Priorities: piece development, central control (e4/e5/d4/d5 squares),
    castling, and avoiding moving the same piece twice without good reason.
    """

    name = "Opening"
    description = "Applies opening principles: development, centre control, and king safety"

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        return f"""You are a chess OPENING expert playing as Black against Stockfish.

Your priorities in this order:
1. Develop knights and bishops towards the centre.
2. Control the central squares e4, e5, d4, d5.
3. Castle early to safeguard your king.
4. Avoid moving the same piece twice unless it wins material.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Identify which pieces are still undeveloped.
- Choose the move that best follows opening principles from the legal list above.
- Give a brief explanation of your choice (1-2 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
