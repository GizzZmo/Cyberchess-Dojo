import chess
from agents.base_agent import BaseChessAgent
from opening_book import FAMOUS_GAMES


class PositionalAgent(BaseChessAgent):
    """
    Specialises in long-range positional chess.

    Evaluates pawn structure, piece activity, weak squares, open files,
    outpost squares, and king safety — improvements that pay off over
    many moves rather than immediately.

    Draws on the positional legacies of Capablanca, Petrosian, Karpov,
    and other masters of strategic chess.
    """

    name = "Positional"
    description = "Evaluates pawn structure, piece activity, weak squares, and long-term strategy"

    # Key positional games that illustrate strategic principles
    _POSITIONAL_REFERENCES: list[str] = (
        FAMOUS_GAMES.get("Queen's Gambit", [])[:2]
        + FAMOUS_GAMES.get("Ruy Lopez", [])[:2]
        + FAMOUS_GAMES.get("Caro-Kann Defense", [])[:1]
    )

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        ref_block = "\n".join(
            f"  • {g}" for g in self._POSITIONAL_REFERENCES[:4]
        )
        return f"""You are a chess POSITIONAL strategist playing as Black against Stockfish.
You have absorbed the positional wisdom of Capablanca, Karpov, Petrosian, and Nimzowitsch,
as demonstrated in landmark games such as:
{ref_block}

Your focus is long-term structural and strategic advantages:
- Pawn structure: avoid isolated, doubled, or backward pawns; create passed pawns.
- Piece activity: place pieces on their optimal squares; improve passive pieces.
- Weak squares: occupy outposts and exploit holes in the opponent's camp.
- Open files: contest them with rooks and the queen.
- King safety: ensure your king is sheltered; expose the enemy king where possible.
- The principle of two weaknesses: create a second front once the first is established.
- Prophylaxis: anticipate and neutralise the opponent's plans (Petrosian's method).

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Assess the pawn structure and identify the key positional imbalance.
- Determine which piece is most in need of improvement.
- Choose the move that makes the greatest long-term positional progress.
- Briefly explain your strategic reasoning, citing a principle if applicable (2-3 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
