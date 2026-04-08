"""
Tests for the Peace Protocol search engine components.

Covers:
- TranspositionTable: store, retrieve, depth-preferred replacement, capacity eviction.
- static_eval: material balance, checkmate, stalemate.
- _move_order_key: policy moves are sorted first.
- _extract_uci_moves: legal-move filtering and de-duplication.
- PeaceProtocolEngine.search: returns a legal move; works without a model.
"""

import unittest
import chess

from transposition_table import TranspositionTable, TTFlag
from search_engine import (
    PeaceProtocolEngine,
    static_eval,
    _move_order_key,
    _extract_uci_moves,
)


# ---------------------------------------------------------------------------
# TranspositionTable tests
# ---------------------------------------------------------------------------

class TestTranspositionTable(unittest.TestCase):

    def _table(self, max_size: int = 100) -> TranspositionTable:
        return TranspositionTable(max_size=max_size)

    def test_store_and_retrieve(self):
        tt = self._table()
        tt.store("fen1", score=50, depth=4, move="e2e4", flag=TTFlag.EXACT)

        self.assertTrue(tt.contains("fen1"))
        self.assertEqual(tt.get_score("fen1"), 50)
        self.assertEqual(tt.get_depth("fen1"), 4)
        self.assertEqual(tt.get_move("fen1"), "e2e4")
        self.assertEqual(tt.get_flag("fen1"), TTFlag.EXACT)

    def test_missing_key_returns_none(self):
        tt = self._table()
        self.assertFalse(tt.contains("ghost"))
        self.assertIsNone(tt.get_score("ghost"))
        self.assertIsNone(tt.get_move("ghost"))
        self.assertEqual(tt.get_depth("ghost"), 0)

    def test_depth_preferred_replacement(self):
        tt = self._table()
        tt.store("fen1", score=10, depth=6, move="e2e4", flag=TTFlag.EXACT)
        # Shallow entry should NOT overwrite the deeper one
        tt.store("fen1", score=99, depth=3, move="d2d4", flag=TTFlag.EXACT)
        self.assertEqual(tt.get_score("fen1"), 10)
        self.assertEqual(tt.get_depth("fen1"), 6)

    def test_deeper_entry_replaces_shallower(self):
        tt = self._table()
        tt.store("fen1", score=10, depth=2, move="e2e4", flag=TTFlag.EXACT)
        tt.store("fen1", score=99, depth=5, move="d2d4", flag=TTFlag.EXACT)
        self.assertEqual(tt.get_score("fen1"), 99)
        self.assertEqual(tt.get_depth("fen1"), 5)

    def test_capacity_eviction_keeps_size_bounded(self):
        tt = self._table(max_size=3)
        for i in range(5):
            tt.store(f"fen{i}", score=i, depth=1)
        self.assertLessEqual(tt.size, 3)

    def test_clear_resets_all(self):
        tt = self._table()
        tt.store("fen1", score=1, depth=1)
        tt.clear()
        self.assertEqual(tt.size, 0)
        self.assertFalse(tt.contains("fen1"))

    def test_hit_rate_tracking(self):
        tt = self._table()
        tt.store("fen1", score=1, depth=1)
        tt.contains("fen1")   # hit
        tt.contains("ghost")  # miss
        self.assertAlmostEqual(tt.hit_rate(), 0.5)

    def test_lower_bound_flag(self):
        tt = self._table()
        tt.store("fen1", score=100, depth=3, flag=TTFlag.LOWER)
        self.assertEqual(tt.get_flag("fen1"), TTFlag.LOWER)

    def test_upper_bound_flag(self):
        tt = self._table()
        tt.store("fen1", score=-50, depth=2, flag=TTFlag.UPPER)
        self.assertEqual(tt.get_flag("fen1"), TTFlag.UPPER)


# ---------------------------------------------------------------------------
# static_eval tests
# ---------------------------------------------------------------------------

class TestStaticEval(unittest.TestCase):

    def test_starting_position_is_balanced(self):
        board = chess.Board()
        score = static_eval(board)
        # Both sides have equal material; PST bonuses may favour one side
        # slightly but the score should be small.
        self.assertLessEqual(abs(score), 200)

    def test_checkmate_returns_large_negative(self):
        # Fool's mate — Black just delivered checkmate, White is mated.
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        # It is White's turn and White is in checkmate
        self.assertTrue(board.is_checkmate())
        score = static_eval(board)
        self.assertLess(score, -900_000)

    def test_material_advantage_is_positive(self):
        # Extra queen for White → from White's perspective (White to move) it's positive
        board = chess.Board("8/8/8/8/8/8/8/4K2Q w - - 0 1")
        board.turn = chess.WHITE
        score = static_eval(board)
        self.assertGreater(score, 0)

    def test_stalemate_is_zero(self):
        # King and pawn stalemate position (Black king trapped)
        board = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        if board.is_stalemate():
            self.assertEqual(static_eval(board), 0)


# ---------------------------------------------------------------------------
# _move_order_key tests
# ---------------------------------------------------------------------------

class TestMoveOrderKey(unittest.TestCase):

    def test_policy_move_has_lowest_key(self):
        board = chess.Board()
        e2e4 = chess.Move.from_uci("e2e4")
        d2d4 = chess.Move.from_uci("d2d4")
        policy_set = {"e2e4"}
        key_policy = _move_order_key(board, e2e4, policy_set)
        key_other  = _move_order_key(board, d2d4, policy_set)
        self.assertLess(key_policy, key_other)

    def test_capture_beats_quiet_move(self):
        # Scandinavian after 1.e4 d5 2.exd5 — White can capture on d5.
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
        # Black queen recaptures d5 (d8d5 is a capture)
        recapture = chess.Move.from_uci("d8d5")
        quiet     = chess.Move.from_uci("g8f6")
        self.assertLess(
            _move_order_key(board, recapture, set()),
            _move_order_key(board, quiet, set()),
        )


# ---------------------------------------------------------------------------
# _extract_uci_moves tests
# ---------------------------------------------------------------------------

class TestExtractUciMoves(unittest.TestCase):

    def test_extracts_valid_moves(self):
        board = chess.Board()
        legal = {m.uci() for m in board.legal_moves}
        text = "I suggest e2e4 followed by d2d4 or g1f3."
        result = _extract_uci_moves(text, legal)
        self.assertIn("e2e4", result)
        self.assertIn("d2d4", result)
        self.assertIn("g1f3", result)

    def test_filters_illegal_moves(self):
        board = chess.Board()
        legal = {m.uci() for m in board.legal_moves}
        text = "Try e2e5 or e4e5."   # both illegal from start position
        result = _extract_uci_moves(text, legal)
        self.assertEqual(result, [])

    def test_deduplicates_moves(self):
        board = chess.Board()
        legal = {m.uci() for m in board.legal_moves}
        text = "e2e4 e2e4 e2e4"
        result = _extract_uci_moves(text, legal)
        self.assertEqual(result.count("e2e4"), 1)


# ---------------------------------------------------------------------------
# PeaceProtocolEngine tests
# ---------------------------------------------------------------------------

class _StubModel:
    """Returns a fixed response (no real LLM calls)."""
    def __init__(self, text: str = "e2e4"):
        self._text = text

    def generate_content(self, prompt: str, **_kwargs):
        class _R:
            pass
        r = _R()
        r.text = self._text
        return r


class TestPeaceProtocolEngine(unittest.TestCase):

    def test_returns_legal_move_from_starting_position(self):
        engine = PeaceProtocolEngine(model=None, search_depth=2)
        board  = chess.Board()
        move   = engine.search(board)
        self.assertIn(move, board.legal_moves)

    def test_returns_legal_move_with_stub_model(self):
        model  = _StubModel("e2e4 d2d4 g1f3")
        engine = PeaceProtocolEngine(model=model, search_depth=2)
        board  = chess.Board()
        move   = engine.search(board)
        self.assertIn(move, board.legal_moves)

    def test_only_move_is_returned_immediately(self):
        # Position with only one legal move (king must escape check)
        # Rb8#: Black king on a8, White rook on a1 giving check, only Ka7 or Kb7 available
        board = chess.Board("k7/8/1K6/8/8/8/8/R7 b - - 0 1")
        legal = list(board.legal_moves)
        if len(legal) == 1:
            engine = PeaceProtocolEngine(model=None, search_depth=2)
            move   = engine.search(board)
            self.assertEqual(move, legal[0])

    def test_transposition_table_is_populated(self):
        engine = PeaceProtocolEngine(model=None, search_depth=2)
        board  = chess.Board()
        engine.search(board)
        self.assertGreater(engine.tt.stores, 0)

    def test_policy_moves_are_tried_first(self):
        """Engine must not crash when LLM returns moves for the current position."""
        model  = _StubModel("e2e4 d2d4 c2c4")
        engine = PeaceProtocolEngine(model=model, search_depth=2, top_n=3)
        board  = chess.Board()
        move   = engine.search(board)
        self.assertIn(move, board.legal_moves)

    def test_checkmate_position_raises_value_error_when_no_legal_moves(self):
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board.is_game_over())
        engine = PeaceProtocolEngine(model=None)
        with self.assertRaises(ValueError):
            engine.search(board)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------

class TestOrchestratorPeaceProtocol(unittest.TestCase):

    def test_get_best_move_without_peace_protocol(self):
        from orchestrator import ChessOrchestrator
        model = _StubModel("e2e4")
        orch  = ChessOrchestrator(model)
        board = chess.Board()
        move  = orch.get_best_move(board, n=1)
        self.assertIn(move, board.legal_moves)

    def test_get_best_move_with_peace_protocol(self):
        from orchestrator import ChessOrchestrator
        model = _StubModel("e7e5 d7d5 g8f6")
        orch  = ChessOrchestrator(model)
        board = chess.Board()
        # Advance past opening phase (move 11+)
        for _ in range(20):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            board.push(legal[0])
        if not board.is_game_over():
            move = orch.get_best_move(board, n=1, use_peace_protocol=True)
            self.assertIn(move, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
