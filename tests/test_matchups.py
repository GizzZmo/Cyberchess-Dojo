import unittest

from cyberchess import _ai_perspective_color, _resolve_matchup


class MatchupTests(unittest.TestCase):
    def test_resolve_matchup_combinations(self):
        self.assertEqual(_resolve_matchup("stockfish-ai"), ("stockfish", "ai"))
        self.assertEqual(_resolve_matchup("stockfish-stockfish"), ("stockfish", "stockfish"))
        self.assertEqual(_resolve_matchup("ai-ai"), ("ai", "ai"))
        self.assertEqual(_resolve_matchup("ai-stockfish"), ("ai", "stockfish"))

    def test_ai_perspective_color_only_for_single_ai_side(self):
        self.assertEqual(_ai_perspective_color("stockfish", "ai"), "black")
        self.assertEqual(_ai_perspective_color("ai", "stockfish"), "white")
        self.assertIsNone(_ai_perspective_color("stockfish", "stockfish"))
        self.assertIsNone(_ai_perspective_color("ai", "ai"))


if __name__ == "__main__":
    unittest.main()
