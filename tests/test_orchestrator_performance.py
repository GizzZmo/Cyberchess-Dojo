import time
import unittest

import chess

from orchestrator import ChessOrchestrator


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FastStubModel:
    def generate_content(self, prompt: str):
        # Always return a legal move from the initial board.
        return _FakeResponse("e2e4")


class OrchestratorPerformanceTests(unittest.TestCase):
    def test_get_best_move_is_fast_with_stub_model(self):
        board = chess.Board()
        model = _FastStubModel()
        orchestrator = ChessOrchestrator(model)

        loops = 40
        t0 = time.perf_counter()
        for _ in range(loops):
            move = orchestrator.get_best_move(board, n=3)
            self.assertIsInstance(move, chess.Move)
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed / loops) * 1000.0

        # Sanity threshold to detect severe regressions in local orchestration logic.
        self.assertLess(avg_ms, 25.0)


if __name__ == "__main__":
    unittest.main()
