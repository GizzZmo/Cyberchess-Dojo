import chess
from agents.base_agent import BaseChessAgent


class EndgameAgent(BaseChessAgent):
    """
    Specialises in endgame technique.

    In endgames, the king becomes a fighting piece, pawn promotion races
    are decisive, and precise technique is required to convert (or hold)
    material advantages.
    """

    name = "Endgame"
    description = "Handles king centralisation, passed pawns, and endgame conversion technique"

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        return f"""You are a chess ENDGAME specialist playing as Black against Stockfish.

Key endgame principles to apply:
- King activity: centralise the king — it is a powerful piece in the endgame.
- Passed pawns: advance them; blockade opponent's passed pawns with the king or pieces.
- Pawn promotion: calculate pawn races accurately, including opposition and key squares.
- Piece coordination: rooks belong behind passed pawns; minor pieces need good outposts.
- Zugzwang: be aware of positions where the side to move is at a disadvantage.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Evaluate which side has better king activity and pawn structure.
- Identify any passed pawns and assess promotion races.
- Choose the move that best applies correct endgame technique.
- Briefly explain your endgame reasoning (2-3 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
