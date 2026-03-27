import chess
from agents.base_agent import BaseChessAgent
import opening_book as ob


class OpeningAgent(BaseChessAgent):
    """
    Specialises in opening play.

    Priorities: piece development, central control (e4/e5/d4/d5 squares),
    castling, and avoiding moving the same piece twice without good reason.

    Enriched with:
    - ECO opening name / code detected from the move history
    - Embedded opening theory: main-line book moves for the current position
    - Polyglot binary opening book support (if a .bin file is available)
    - Curated historical game references for the identified opening
    """

    name = "Opening"
    description = "Applies opening principles with ECO theory and historical game knowledge"

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        # --- Opening knowledge context ---
        eco_code = ob.get_eco_code(board) or "?"
        opening_name = ob.get_opening_name(board) or "Unknown Opening"

        # Book moves: Polyglot first, fall back to embedded theory
        book_moves = ob.get_book_moves(board)
        if not book_moves:
            book_moves = ob.get_theoretic_moves(board)
        # Filter to legal moves (safety check)
        legal_set = set(legal_moves)
        book_moves = [m for m in book_moves if m in legal_set]

        related_games = ob.get_related_games(opening_name)

        # Build context blocks only when data is available
        book_section = ""
        if book_moves:
            book_section = (
                f"\nOpening theory — main-line moves for Black in this position:\n"
                f"  {', '.join(book_moves)}\n"
                "(These are the most theoretically tested responses; prefer them unless "
                "you have a concrete reason to deviate.)\n"
            )

        history_section = ""
        if related_games:
            sample = related_games[:3]
            history_section = (
                "\nHistorical game references for this opening:\n"
                + "\n".join(f"  • {g}" for g in sample)
                + "\n"
            )

        return f"""You are a chess OPENING expert playing as Black against Stockfish.
You have encyclopaedic knowledge of all classical and modern openings (ECO codes A00–E99)
and have studied thousands of historical grandmaster games.

Current opening: {opening_name} (ECO {eco_code})
{book_section}{history_section}
Your priorities in this order:
1. Develop knights and bishops towards the centre.
2. Control the central squares e4, e5, d4, d5.
3. Castle early to safeguard your king.
4. Avoid moving the same piece twice unless it wins material.
5. Follow established opening theory where possible.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Identify which pieces are still undeveloped.
- Consider any opening theory moves listed above first.
- Choose the move that best follows opening principles and theory from the legal list.
- Give a brief explanation citing opening theory if relevant (1-2 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
