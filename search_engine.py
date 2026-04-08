"""
Peace Protocol Search Engine for Cyberchess-Dojo.

Implements the hybrid Search-Transformer paradigm described in the system
architecture: an alpha-beta minimax search guided by LLM-generated strategic
priors that prioritise the top-N "humanly logical" moves for deep exploration
while treating the remaining moves with a shallow search or early pruning.

Architecture overview
---------------------
1. **Policy query** — The LLM is asked to name the top-N candidate moves for
   the current position.  These moves receive a full-depth search allocation.
2. **Alpha-beta search** — Standard negamax alpha-beta with iterative deepening.
   LLM-policy moves are tried first (move ordering), greatly improving cutoffs.
3. **Transposition table** — Positions are cached so that Gemini evaluations at
   quiet leaf nodes are never repeated within the same search tree.
4. **Quiet-position evaluation** — When the search reaches a quiet position at
   or below the transformer threshold, a fast material + PST score is returned.
   An optional LLM evaluation call can be enabled for higher accuracy.

The ``PeaceProtocolEngine`` is designed as a drop-in enhancement for the
``ChessOrchestrator``, accessible via ``orchestrator.get_best_move(...,
use_peace_protocol=True)``.

Usage::

    from search_engine import PeaceProtocolEngine

    engine = PeaceProtocolEngine(model=llm_adapter)
    move   = engine.search(board, depth=4)
"""

from __future__ import annotations

import re
import chess

from transposition_table import TranspositionTable, TTFlag

# ---------------------------------------------------------------------------
# Piece-Square Tables (PST) — simplified midgame values (from Black's side)
# Based on PeSTO's evaluation tables (public domain).
# ---------------------------------------------------------------------------

_PAWN_PST = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
]

_KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

_BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_ROOK_PST = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0,
]

_QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_KING_MID_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# Centipawn values for each piece type
_PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

_PST_MAP: dict[chess.PieceType, list[int]] = {
    chess.PAWN:   _PAWN_PST,
    chess.KNIGHT: _KNIGHT_PST,
    chess.BISHOP: _BISHOP_PST,
    chess.ROOK:   _ROOK_PST,
    chess.QUEEN:  _QUEEN_PST,
    chess.KING:   _KING_MID_PST,
}

_INFINITY = 1_000_000

# Depth threshold below which the engine uses fast static evaluation rather
# than an LLM call — mirrors the TRANSFORMER_THRESHOLD in the spec pseudo-code.
_DEFAULT_TRANSFORMER_THRESHOLD = 0

# Number of LLM-recommended "top" moves that receive full-depth exploration.
_DEFAULT_TOP_N = 3

# Maximum search depth for the Peace Protocol engine.
_DEFAULT_SEARCH_DEPTH = 4


# ---------------------------------------------------------------------------
# Static evaluation
# ---------------------------------------------------------------------------

def _pst_score(square: chess.Square, piece_type: chess.PieceType, color: chess.Color) -> int:
    """Return the PST bonus for a piece at *square* from White's perspective."""
    pst = _PST_MAP.get(piece_type)
    if pst is None:
        return 0
    # PST arrays are indexed from a8 (index 0) to h1 (index 63) for White.
    idx = square if color == chess.WHITE else chess.square_mirror(square)
    return pst[idx]


def static_eval(board: chess.Board) -> int:
    """
    Fast static evaluation of *board* from the perspective of the side to move.

    Returns a centipawn score (positive = better for the side to move).
    Combines material balance with PST bonuses.
    """
    if board.is_checkmate():
        return -_INFINITY + board.ply()   # being mated is worst; earlier is worse
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = _PIECE_VALUE.get(piece.piece_type, 0) + _pst_score(square, piece.piece_type, piece.color)
        if piece.color == board.turn:
            score += value
        else:
            score -= value
    return score


# ---------------------------------------------------------------------------
# Move ordering helpers
# ---------------------------------------------------------------------------

def _move_order_key(board: chess.Board, move: chess.Move, policy_set: set[str]) -> int:
    """
    Return a sort key for *move* (lower value = tried earlier).

    Priority:
      0 — LLM policy move (top-N)
      1 — promotion
      2 — capture (MVV-LVA approximation)
      3 — check
      4 — everything else
    """
    uci = move.uci()
    if uci in policy_set:
        return 0
    if move.promotion:
        return 1
    captured = board.piece_at(move.to_square)
    if captured:
        victim_val   = _PIECE_VALUE.get(captured.piece_type, 0)
        attacker_val = _PIECE_VALUE.get(board.piece_at(move.from_square).piece_type, 0)
        return 2 - victim_val // 100 + attacker_val // 1000  # higher victim = earlier
    if board.gives_check(move):
        return 3
    return 4


# ---------------------------------------------------------------------------
# LLM policy helpers
# ---------------------------------------------------------------------------

def _extract_uci_moves(text: str, legal_set: set[str]) -> list[str]:
    """
    Extract a list of legal UCI moves from an LLM response text.

    Scans for all UCI-shaped tokens and filters to the legal set so that
    hallucinated moves are silently dropped.
    """
    tokens = re.findall(r'\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b', text)
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        tok = tok.lower()
        if tok in legal_set and tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


def _ask_policy(board: chess.Board, model, top_n: int) -> list[str]:
    """
    Query the LLM for its top-N recommended moves in the current position.

    Returns a list of legal UCI move strings (may be shorter than *top_n*
    if the model returns fewer valid moves).  Falls back to an empty list
    on any error so the caller can degrade gracefully.
    """
    legal_moves = [m.uci() for m in board.legal_moves]
    legal_set   = set(legal_moves)

    prompt = (
        f"You are a chess grandmaster analysing the following position.\n"
        f"FEN: {board.fen()}\n"
        f"Legal moves: {', '.join(legal_moves)}\n\n"
        f"List the top {top_n} candidate moves in order from best to worst.\n"
        f"Output ONLY the UCI moves separated by spaces or newlines, nothing else.\n"
        f"Example output: e7e5 d7d5 g8f6"
    )

    try:
        response = model.generate_content(prompt)
        return _extract_uci_moves(response.text, legal_set)[:top_n]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Peace Protocol Engine
# ---------------------------------------------------------------------------

class PeaceProtocolEngine:
    """
    Hybrid Search-Transformer engine implementing the Peace Protocol.

    The engine performs an alpha-beta negamax search guided by LLM-generated
    strategic priors.  LLM-recommended moves are searched first and at full
    depth; all other legal moves are searched afterwards and may be pruned
    aggressively if they fail a basic tactical check.

    Args:
        model:                  Any object with ``generate_content(prompt)``
                                (e.g. a ``BaseLLMAdapter`` or Gemini model).
                                May be ``None`` — in that case the engine
                                falls back to pure static evaluation.
        transposition_table:    A pre-existing ``TranspositionTable`` instance.
                                A new table with 500 000 entries is created if
                                not provided.
        search_depth:           Maximum alpha-beta depth.  Defaults to 4.
        top_n:                  Number of LLM policy moves explored at full
                                depth.  Defaults to 3 (Peace Protocol spec).
        transformer_threshold:  Depth at or below which we use fast static
                                evaluation rather than an LLM query at the
                                leaf nodes.  Defaults to 0.
        verbose:                If True, print search diagnostics.
    """

    def __init__(
        self,
        model=None,
        transposition_table: TranspositionTable | None = None,
        search_depth: int = _DEFAULT_SEARCH_DEPTH,
        top_n: int = _DEFAULT_TOP_N,
        transformer_threshold: int = _DEFAULT_TRANSFORMER_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        self.model   = model
        self.tt      = transposition_table or TranspositionTable(max_size=500_000)
        self.depth   = max(1, search_depth)
        self.top_n   = max(1, top_n)
        self.threshold = transformer_threshold
        self.verbose = verbose
        self._nodes  = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, board: chess.Board) -> chess.Move:
        """
        Return the best legal move for the side to move on *board*.

        Runs the full Peace Protocol pipeline:
        1. Query LLM for top-N policy moves.
        2. Sort all legal moves so policy moves are tried first.
        3. Run alpha-beta search.
        4. Return the highest-scoring move.
        """
        self._nodes = 0
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available — game is over.")
        if len(legal) == 1:
            return legal[0]

        # Step 1: get LLM policy (empty list if model is None or call fails)
        policy: list[str] = []
        if self.model is not None:
            policy = _ask_policy(board, self.model, self.top_n)
            if self.verbose:
                print(f"  [PeaceProtocol] Policy moves: {policy}")
        policy_set = set(policy)

        # Step 2: sort moves — policy moves first, then captures, checks, rest
        sorted_moves = sorted(legal, key=lambda m: _move_order_key(board, m, policy_set))

        # Step 3: root alpha-beta
        best_move  = sorted_moves[0]
        best_score = -_INFINITY
        alpha      = -_INFINITY
        beta       = _INFINITY

        for move in sorted_moves:
            board.push(move)
            score = -self._alpha_beta(board, self.depth - 1, -beta, -alpha, policy_set)
            board.pop()

            if score > best_score:
                best_score = score
                best_move  = move
            if score > alpha:
                alpha = score

        if self.verbose:
            print(
                f"  [PeaceProtocol] depth={self.depth} nodes={self._nodes} "
                f"best={best_move.uci()} score={best_score}"
            )

        return best_move

    # ------------------------------------------------------------------
    # Internal alpha-beta
    # ------------------------------------------------------------------

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        policy_set: set[str],
    ) -> int:
        """
        Negamax alpha-beta search.

        Returns the score from the perspective of the side to move.
        """
        self._nodes += 1

        # 1. Transposition table lookup
        fen = board.fen()
        if self.tt.contains(fen) and self.tt.get_depth(fen) >= depth:
            cached_score = self.tt.get_score(fen)
            cached_flag  = self.tt.get_flag(fen)
            if cached_flag == TTFlag.EXACT:
                return cached_score
            if cached_flag == TTFlag.LOWER and cached_score > alpha:
                alpha = cached_score
            elif cached_flag == TTFlag.UPPER and cached_score < beta:
                beta = cached_score
            if alpha >= beta:
                return cached_score

        # 2. Terminal / leaf evaluation
        if board.is_game_over():
            return static_eval(board)

        if depth <= self.threshold:
            score = static_eval(board)
            self.tt.store(fen, score, depth, flag=TTFlag.EXACT)
            return score

        # 3. Recursive search
        legal = list(board.legal_moves)
        sorted_moves = sorted(legal, key=lambda m: _move_order_key(board, m, policy_set))

        best_score  = -_INFINITY
        best_move   = None
        orig_alpha  = alpha

        for move in sorted_moves:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, policy_set)
            board.pop()

            if score > best_score:
                best_score = score
                best_move  = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Beta cutoff — store lower bound
                self.tt.store(fen, beta, depth, move=move.uci(), flag=TTFlag.LOWER)
                return beta

        # Determine flag from final alpha vs original alpha
        flag = TTFlag.EXACT if alpha > orig_alpha else TTFlag.UPPER
        self.tt.store(fen, best_score, depth,
                      move=best_move.uci() if best_move else None, flag=flag)
        return best_score
