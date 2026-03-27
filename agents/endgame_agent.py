import chess
from agents.base_agent import BaseChessAgent
from opening_book import FAMOUS_GAMES


class EndgameAgent(BaseChessAgent):
    """
    Specialises in endgame technique.

    In endgames, the king becomes a fighting piece, pawn promotion races
    are decisive, and precise technique is required to convert (or hold)
    material advantages.

    Draws on the endgame mastery of Capablanca, Rubinstein, and Fischer.
    """

    name = "Endgame"
    description = "Handles king centralisation, passed pawns, and endgame conversion technique"

    # Famous endgame studies and games used as reference
    _ENDGAME_REFERENCES: list[str] = FAMOUS_GAMES.get("Endgame Masterpieces", [])

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        ref_block = "\n".join(
            f"  • {g}" for g in self._ENDGAME_REFERENCES[:4]
        )
        return f"""You are a chess ENDGAME specialist playing as Black against Stockfish.
You have mastered all classical endgame theory and studied the landmark endgames of history:
{ref_block}

Key endgame principles to apply:
- King activity: centralise the king — it is a powerful piece in the endgame.
- Passed pawns: advance them; blockade opponent's passed pawns with the king or pieces.
- Pawn promotion: calculate pawn races accurately, including opposition and key squares.
- Piece coordination: rooks belong behind passed pawns; minor pieces need good outposts.
- Zugzwang: be aware of positions where the side to move is at a disadvantage.
- Lucena and Philidor positions: apply the correct technique in rook endgames.
- Bishop vs knight: exploit the long-range power of the bishop in open positions.
- The opposition: use king opposition to control key squares in pawn endings.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Evaluate which side has better king activity and pawn structure.
- Identify any passed pawns and assess promotion races.
- Choose the move that best applies correct endgame technique.
- Briefly explain your endgame reasoning, citing a principle or known technique (2-3 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
