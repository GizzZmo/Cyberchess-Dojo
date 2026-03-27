import chess
from agents.base_agent import BaseChessAgent
from opening_book import FAMOUS_GAMES


class TacticalAgent(BaseChessAgent):
    """
    Specialises in short-range tactics.

    Looks for checks, captures, forks (attacking two pieces at once),
    pins, skewers, and other forcing sequences that win material or
    deliver checkmate.

    Informed by famous historical games that showcase recurring tactical
    motifs at the grandmaster level.
    """

    name = "Tactical"
    description = "Finds checks, captures, forks, pins, skewers, and forcing sequences"

    # Representative tactical masterpieces used as in-prompt examples
    _TACTICAL_REFERENCES: list[str] = (
        FAMOUS_GAMES.get("Tactical Masterpieces", [])
        + FAMOUS_GAMES.get("King's Gambit", [])[:2]
    )

    def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
        ref_block = "\n".join(
            f"  • {g}" for g in self._TACTICAL_REFERENCES[:5]
        )
        return f"""You are a chess TACTICAL specialist playing as Black against Stockfish.
You have studied all famous tactical masterpieces in chess history, including:
{ref_block}

Your sole focus is forcing moves and material gain:
- Checks that lead to checkmate or win material.
- Captures of undefended or overloaded pieces.
- Forks (one piece attacks two enemy pieces simultaneously).
- Pins and skewers that win material.
- Quiet moves that set up an unavoidable tactical threat.

Recall the canonical tactical patterns: back-rank weakness, discovered check,
zwischenzug (in-between move), deflection, decoy, interference, and x-ray attack.

Current board (FEN): {board.fen()}
Legal moves available: {', '.join(legal_moves)}

Instructions:
- Scan for all checks and captures in the legal move list.
- Identify the most forcing / highest-value tactical sequence.
- If no immediate tactic exists, choose the move that creates the biggest threat.
- Briefly explain your tactical reasoning, citing a pattern if applicable (2-3 sentences).
- On the very last line of your response, write ONLY the UCI move (e.g. e7e5).
"""
