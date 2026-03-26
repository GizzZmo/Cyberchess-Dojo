import chess
from agents.base_agent import BaseChessAgent


class PositionalAgent(BaseChessAgent):
    """
    Specialises in long-range positional chess.

    Evaluates pawn structure, piece activity, weak squares, open files,
    outpost squares, and king safety — improvements that pay off over
    many moves rather than immediately.
    """

    name = "Positional"
    description = "Evaluates pawn structure, piece activity, weak squares, and long-term strategy"

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        return f"""You are a chess POSITIONAL strategist playing as Black against Stockfish.

Your focus is long-term structural and strategic advantages:
- Pawn structure: avoid isolated, doubled, or backward pawns; create passed pawns.
- Piece activity: place pieces on their optimal squares; improve passive pieces.
- Weak squares: occupy outposts and exploit holes in the opponent's camp.
- Open files: contest them with rooks and the queen.
- King safety: ensure your king is sheltered; expose the enemy king where possible.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Assess the pawn structure and identify the key positional imbalance.
- Determine which piece is most in need of improvement.
- Choose the move that makes the greatest long-term positional progress.
- Briefly explain your strategic reasoning (2-3 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
