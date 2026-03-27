import os
import tempfile
import unittest

from adaptive_system import AdaptiveTrainingManager


class AdaptiveSystemTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()

    def tearDown(self):
        try:
            os.remove(self._tmp.name)
        except OSError:
            pass

    def test_recovery_regime_when_recent_score_is_low(self):
        manager = AdaptiveTrainingManager(filepath=self._tmp.name)
        history = [{"result": "1-0", "ai_color": "black"} for _ in range(8)]

        plan = manager.plan_next_game(
            base_skill=10,
            base_time=0.1,
            base_best_of_n=3,
            elo_history=history,
        )

        self.assertEqual(plan["regime"], "recovery")
        self.assertEqual(plan["stockfish_skill"], 9)
        self.assertEqual(plan["best_of_n"], 4)
        self.assertLess(plan["stockfish_time"], 0.1)

    def test_challenge_regime_when_recent_score_is_high(self):
        manager = AdaptiveTrainingManager(filepath=self._tmp.name)
        history = [{"result": "0-1", "ai_color": "black"} for _ in range(8)]

        plan = manager.plan_next_game(
            base_skill=10,
            base_time=0.1,
            base_best_of_n=3,
            elo_history=history,
        )

        self.assertEqual(plan["regime"], "challenge")
        self.assertEqual(plan["stockfish_skill"], 11)
        self.assertEqual(plan["best_of_n"], 4)
        self.assertGreater(plan["stockfish_time"], 0.1)


if __name__ == "__main__":
    unittest.main()
