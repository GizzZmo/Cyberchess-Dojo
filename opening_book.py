"""
Opening book and historical games knowledge base for Cyberchess-Dojo.

Provides:
- ECO opening name / code lookup from the board's move history
- Embedded opening theory: main-line responses for Black in ~50 key positions
- Curated historical games database: ~100 famous games organised by opening
- Optional Polyglot binary opening book support (chess.polyglot)

Public API
----------
get_opening_name(board)       -> str | None
get_eco_code(board)           -> str | None
get_theoretic_moves(board)    -> list[str]  (UCI, filtered to legal moves)
get_related_games(name)       -> list[str]  (human-readable game descriptions)
get_book_moves(board, path)   -> list[str]  (UCI, from Polyglot .bin if available)
"""

import os
import chess
import chess.polyglot

# ---------------------------------------------------------------------------
# ECO opening database
# Maps a space-separated UCI move sequence to (eco_code, opening_name).
# The sequence ends with White's last move; the position is therefore Black's
# turn (or the very first move of the game before any moves are played).
# ---------------------------------------------------------------------------

ECO_OPENINGS: dict[str, tuple[str, str]] = {
    # ---- A-series: flank / irregular openings --------------------------------
    "": ("A00", "Starting Position"),
    "g1f3": ("A04", "Réti Opening"),
    "g1f3 d7d5": ("A06", "Réti Opening"),
    "g1f3 d7d5 c2c4": ("A11", "Réti: King's Indian Attack"),
    "g1f3 g8f6 c2c4 g7g6": ("A14", "English: Neo-Catalan"),
    "c2c4": ("A10", "English Opening"),
    "c2c4 e7e5": ("A20", "English: 1...e5"),
    "c2c4 e7e5 g1f3": ("A22", "English: Three Knights"),
    "c2c4 e7e5 g1f3 b8c6": ("A25", "English: Closed"),
    "c2c4 g8f6": ("A15", "English: Anglo-Indian"),
    "c2c4 g8f6 g1f3 c7c5": ("A34", "English: Symmetrical"),
    "c2c4 c7c5": ("A30", "English: Symmetrical Defense"),
    "c2c4 g7g6": ("A10", "English: Anglo-Grünfeld"),
    "c2c4 g8f6 b1c3 d7d5": ("A13", "English: Agincourt Defense"),
    "b2b4": ("A00", "Sokolsky Opening (Orangutan)"),
    "f2f4": ("A02", "Bird's Opening"),
    "f2f4 e7e5": ("A02", "Bird's Opening: From's Gambit"),
    "f2f4 d7d5 g1f3 g8f6": ("A03", "Bird's Opening: Dutch Variation"),
    "g2g4": ("A00", "Grob's Attack"),
    "d2d4 f7f5": ("A80", "Dutch Defense"),
    "d2d4 f7f5 g2g3": ("A81", "Dutch: Leningrad Var."),
    "d2d4 f7f5 c2c4": ("A85", "Dutch: Classical/Leningrad"),
    "d2d4 f7f5 c2c4 g8f6 g1f3 e7e6": ("A92", "Dutch: Classical"),
    "d2d4 f7f5 c2c4 g8f6 g2g3 g7g6 g1f3 f8g7": ("A81", "Dutch: Leningrad"),

    # ---- B-series: semi-open games (e4, not e5) -----------------------------
    "e2e4": ("B00", "King's Pawn Opening"),
    "e2e4 c7c5": ("B20", "Sicilian Defense"),
    "e2e4 c7c5 b1c3": ("B20", "Sicilian: Closed"),
    "e2e4 c7c5 c2c3": ("B22", "Sicilian: Alapin Variation"),
    "e2e4 c7c5 f2f4": ("B21", "Sicilian: Grand Prix Attack"),
    "e2e4 c7c5 g1f3": ("B40", "Sicilian: Open"),
    "e2e4 c7c5 g1f3 d7d6": ("B70", "Sicilian: Classical"),
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4": ("B75", "Sicilian: Dragon Var."),
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6": ("B90", "Sicilian: Najdorf"),
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6": ("B70", "Sicilian: Dragon"),
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 b8c6": ("B56", "Sicilian: Classical"),
    "e2e4 c7c5 g1f3 b8c6": ("B40", "Sicilian: Four Knights setup"),
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4": ("B57", "Sicilian: Classical"),
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6": ("B72", "Sicilian: Dragon"),
    "e2e4 c7c5 g1f3 e7e6": ("B40", "Sicilian: Kan/Taimanov setup"),
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6": ("B43", "Sicilian: Kan"),
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6": ("B47", "Sicilian: Taimanov"),
    "e2e4 c7c6": ("B10", "Caro-Kann Defense"),
    "e2e4 c7c6 d2d4 d7d5": ("B13", "Caro-Kann: Main Line"),
    "e2e4 c7c6 d2d4 d7d5 b1c3": ("B15", "Caro-Kann: Classical"),
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4": ("B18", "Caro-Kann: Classical Main Line"),
    "e2e4 c7c6 d2d4 d7d5 e4d5": ("B13", "Caro-Kann: Exchange Variation"),
    "e2e4 c7c6 d2d4 d7d5 e4e5": ("B12", "Caro-Kann: Advance Variation"),
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5": ("B12", "Caro-Kann: Advance, Short Variation"),
    "e2e4 d7d5": ("B01", "Scandinavian Defense"),
    "e2e4 d7d5 e4d5": ("B01", "Scandinavian: Main Line"),
    "e2e4 d7d5 e4d5 d8d5": ("B01", "Scandinavian: Center Counter"),
    "e2e4 d7d5 e4d5 g8f6": ("B01", "Scandinavian: Modern"),
    "e2e4 d7d6": ("B07", "Pirc Defense"),
    "e2e4 d7d6 d2d4 g8f6": ("B07", "Pirc: Classical Setup"),
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6": ("B08", "Pirc: Classical"),
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f1e2 f8g7": ("B09", "Pirc: Austrian Attack setup"),
    "e2e4 g7g6": ("B06", "Modern Defense"),
    "e2e4 g7g6 d2d4 d7d6": ("B06", "Modern/Pirc"),
    "e2e4 b8c6": ("B00", "Nimzowitsch Defense"),
    "e2e4 g8f6": ("B02", "Alekhine's Defense"),
    "e2e4 g8f6 e4e5 f6d5": ("B03", "Alekhine's Defense: Main Line"),
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6": ("B04", "Alekhine's Defense: Modern"),

    # ---- C-series: French + open games (e4 e5) ------------------------------
    "e2e4 e7e6": ("C00", "French Defense"),
    "e2e4 e7e6 d2d4 d7d5": ("C01", "French Defense: Main Line"),
    "e2e4 e7e6 d2d4 d7d5 e4e5": ("C02", "French: Advance Variation"),
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5": ("C02", "French: Advance, Main Line"),
    "e2e4 e7e6 d2d4 d7d5 b1c3": ("C10", "French: Rubinstein/Winawer"),
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4": ("C15", "French: Winawer Variation"),
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6": ("C11", "French: Classical Variation"),
    "e2e4 e7e6 d2d4 d7d5 b1d2": ("C11", "French: Tarrasch Variation"),
    "e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 g1f3 g8f6": ("C01", "French: Exchange"),
    "e2e4 e7e5": ("C20", "Open Game"),
    "e2e4 e7e5 g1f3": ("C40", "King's Knight Opening"),
    "e2e4 e7e5 g1f3 b8c6": ("C44", "Three Knights / Four Knights setup"),
    "e2e4 e7e5 g1f3 b8c6 f1b5": ("C60", "Ruy Lopez / Spanish Game"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6": ("C68", "Ruy Lopez: Morphy Defense"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4": ("C70", "Ruy Lopez: Main Line"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6": ("C78", "Ruy Lopez: Closed/Open"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1": ("C80", "Ruy Lopez: Open Variation"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f6e4": ("C83", "Ruy Lopez: Open, 9.d3"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7": ("C84", "Ruy Lopez: Closed"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6": ("C65", "Ruy Lopez: Berlin Defense"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 f8c5": ("C61", "Ruy Lopez: Bird's Defense"),
    "e2e4 e7e5 g1f3 b8c6 f1b5 d7d6": ("C70", "Ruy Lopez: Steinitz Defense"),
    "e2e4 e7e5 g1f3 b8c6 f1c4": ("C50", "Italian Game"),
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5": ("C50", "Giuoco Piano"),
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3": ("C54", "Giuoco Piano: Main Line"),
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4": ("C51", "Evans Gambit"),
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6": ("C55", "Two Knights Defense"),
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d4": ("C55", "Two Knights: Modern Attack"),
    "e2e4 e7e5 g1f3 b8c6 d2d4": ("C44", "Scotch Game"),
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4": ("C44", "Scotch Game: Main Line"),
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4": ("C45", "Scotch Game"),
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6": ("C45", "Scotch: Schmidt Variation"),
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5": ("C45", "Scotch: Classical Var."),
    "e2e4 e7e5 g1f3 g8f6": ("C42", "Petrov's Defense"),
    "e2e4 e7e5 g1f3 g8f6 f3e5": ("C42", "Petrov: Classical Attack"),
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4": ("C42", "Petrov: Main Line"),
    "e2e4 e7e5 g1f3 d7d6": ("C41", "Philidor Defense"),
    "e2e4 e7e5 f2f4": ("C33", "King's Gambit"),
    "e2e4 e7e5 f2f4 e5f4": ("C33", "King's Gambit Accepted"),
    "e2e4 e7e5 f2f4 e5f4 g1f3": ("C37", "King's Gambit: Modern Defense"),
    "e2e4 e7e5 f2f4 d7d5": ("C31", "King's Gambit: Falkbeer Counter-Gambit"),
    "e2e4 e7e5 f2f4 f8c5": ("C30", "King's Gambit Declined"),
    "e2e4 e7e5 d2d4": ("C21", "Center Game"),
    "e2e4 e7e5 b1c3": ("C25", "Vienna Game"),
    "e2e4 e7e5 b1c3 b8c6": ("C26", "Vienna Game: Main Line"),
    "e2e4 e7e5 b1c3 g8f6": ("C26", "Vienna Gambit"),
    "e2e4 e7e5 g1f3 b8c6 b1c3": ("C46", "Three Knights Game"),
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6": ("C47", "Four Knights Game"),
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1b5": ("C48", "Four Knights: Spanish"),
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1c4": ("C50", "Four Knights: Italian"),

    # ---- D-series: closed games (d4 d5) -------------------------------------
    "d2d4": ("D00", "Queen's Pawn Opening"),
    "d2d4 d7d5": ("D00", "Closed Game"),
    "d2d4 d7d5 c1f4": ("D00", "London System"),
    "d2d4 d7d5 g1f3 g8f6 c1f4": ("D02", "London System: Main"),
    "d2d4 d7d5 e2e3": ("D00", "Colle System"),
    "d2d4 d7d5 b1c3": ("D00", "Veresov Attack"),
    "d2d4 d7d5 c2c4": ("D06", "Queen's Gambit"),
    "d2d4 d7d5 c2c4 d5c4": ("D20", "Queen's Gambit Accepted"),
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6": ("D23", "QGA: Main Line"),
    "d2d4 d7d5 c2c4 e7e6": ("D30", "Queen's Gambit Declined"),
    "d2d4 d7d5 c2c4 e7e6 b1c3": ("D30", "QGD: Orthodox Defense"),
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6": ("D40", "QGD: Orthodox"),
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5": ("D55", "QGD: Classical"),
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8": ("D58", "QGD: Tartakower"),
    "d2d4 d7d5 c2c4 e7e6 g1f3": ("D30", "QGD: Main Line"),
    "d2d4 d7d5 c2c4 c7c6": ("D10", "Slav Defense"),
    "d2d4 d7d5 c2c4 c7c6 g1f3": ("D11", "Slav: Main Line"),
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6": ("D15", "Slav: Accepted/Czech"),
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 g1f3 e7e6": ("D46", "Semi-Slav Defense"),
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 g1f3 e7e6 e2e3": ("D47", "Semi-Slav: Meran"),
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 g1f3 e7e6 c1g5": ("D43", "Semi-Slav: Moscow"),
    "d2d4 d7d5 c2c4 e7e5": ("D40", "Albin Counter-Gambit"),
    "d2d4 e7e6": ("A40", "Queen's Pawn: Horwitz Defense"),
    "d2d4 e7e5": ("A40", "Englund Gambit"),
    "d2d4 c7c5": ("A41", "Modern Defense: Queen's Pawn"),

    # ---- E-series: Indian defenses ------------------------------------------
    "d2d4 g8f6": ("E00", "Indian Defenses"),
    "d2d4 g8f6 c2c4": ("E00", "Indian Defense"),
    "d2d4 g8f6 c2c4 g7g6": ("E60", "King's Indian Defense"),
    "d2d4 g8f6 c2c4 g7g6 b1c3": ("E61", "KID: Main Line"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7": ("E62", "KID: Classical"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4": ("E91", "KID: Orthodox"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3": ("E92", "KID: Classical"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8": ("E92", "KID: Classical Main"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2": ("E94", "KID: Classical Var."),
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f4": ("E97", "KID: Four Pawns Attack"),
    "d2d4 g8f6 c2c4 g7g6 g1f3 f8g7 g2g3": ("E60", "KID: Fianchetto Var."),
    "d2d4 g8f6 c2c4 e7e6": ("E00", "Nimzo/Queen's Indian/Bogo-Indian"),
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4": ("E20", "Nimzo-Indian Defense"),
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3": ("E40", "Nimzo-Indian: Rubinstein"),
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2": ("E35", "Nimzo-Indian: Classical"),
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 a2a3": ("E28", "Nimzo-Indian: Sämisch"),
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 g1f3": ("E48", "Nimzo-Indian: Rubinstein"),
    "d2d4 g8f6 c2c4 e7e6 g1f3": ("E10", "Queen's Indian / Bogo-Indian"),
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6": ("E12", "Queen's Indian Defense"),
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 b1c3 c8b7": ("E15", "Queen's Indian: Main Line"),
    "d2d4 g8f6 c2c4 e7e6 g1f3 f8b4": ("E11", "Bogo-Indian Defense"),
    "d2d4 g8f6 c2c4 c7c5": ("E00", "Benoni Defense"),
    "d2d4 g8f6 c2c4 c7c5 d4d5": ("A60", "Benoni: Modern"),
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6": ("A61", "Benoni: Modern Main"),
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6": ("A61", "Benoni: Classical"),
    "d2d4 g8f6 c2c4 d7d5": ("D04", "Queen's Gambit"),
    "d2d4 g8f6 g1f3 g7g6": ("A48", "King's Indian Attack"),
    "d2d4 g8f6 g1f3 d7d5 c1g5": ("D01", "Richter-Veresov Attack"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5": ("E60", "Grünfeld Defense setup"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3": ("D85", "Grünfeld Defense"),
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 g1f3 f8g7 d1b3": ("D97", "Grünfeld: Russian System"),
}

# ---------------------------------------------------------------------------
# Opening theory: recommended Black responses at key branch points
# Key = exact UCI move sequence so far (board.move_stack as UCI, joined by spaces)
# Value = ordered list of main-line moves for Black (best first)
# ---------------------------------------------------------------------------

THEORY: dict[str, list[str]] = {
    # --- After White's first move ---
    "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g7g6", "d7d6", "g8f6"],
    "d2d4": ["d7d5", "g8f6", "f7f5", "e7e6", "c7c5"],
    "c2c4": ["e7e5", "g8f6", "c7c5", "g7g6", "e7e6"],
    "g1f3": ["d7d5", "g8f6", "c7c5", "g7g6"],

    # --- Ruy Lopez main lines ---
    "e2e4 e7e5 g1f3 b8c6 f1b5": ["a7a6", "g8f6", "d7d6", "f7f5"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4": ["g8f6", "b7b5", "d7d6"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1": ["f6e4", "f8e7", "b7b5"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 d1e2": ["b7b5", "d7d6"],
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6": ["f6e4", "d7d6"],  # Berlin Defense

    # --- Italian Game ---
    "e2e4 e7e5 g1f3 b8c6 f1c4": ["f8c5", "g8f6", "f8e7"],
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3": ["g8f6", "d7d6", "a7a6"],
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4": ["f8b4", "c5b6", "c5d6"],  # Evans Gambit
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d4": ["e5d4", "d7d6", "f6e4"],  # Two Knights

    # --- Scotch Game ---
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4": ["g8f6", "f8c5", "d8h4"],
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 b1c3": ["f8b4", "f8c5", "d7d5"],

    # --- King's Gambit ---
    "e2e4 e7e5 f2f4": ["e5f4", "d7d5", "f8c5"],
    "e2e4 e7e5 f2f4 e5f4 g1f3": ["g7g5", "d7d5", "g8f6"],

    # --- Petrov's Defense ---
    "e2e4 e7e5 g1f3 g8f6 f3e5": ["d7d6", "b8c6"],
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4": ["d6d5", "f8e7"],

    # --- Sicilian Defense ---
    "e2e4 c7c5 g1f3": ["d7d6", "b8c6", "e7e6", "a7a6"],
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4": ["g8f6", "a7a6", "b8c6"],
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3": ["a7a6", "g7g6", "e7e5"],
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4": ["g7g6", "e7e6", "g8f6"],
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4": ["a7a6", "b8c6", "g8f6"],
    "e2e4 c7c5 c2c3": ["g8f6", "d7d5", "e7e6"],  # Alapin: main responses

    # --- French Defense ---
    "e2e4 e7e6 d2d4 d7d5 b1c3": ["g8f6", "f8b4", "d5e4"],
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4": ["g8f6", "d8d7", "c7c5"],  # Winawer
    "e2e4 e7e6 d2d4 d7d5 e4e5": ["c7c5", "g8e7", "b8d7"],  # Advance

    # --- Caro-Kann Defense ---
    "e2e4 c7c6 d2d4 d7d5 b1c3": ["d5e4", "g8f6", "e7e6"],
    "e2e4 c7c6 d2d4 d7d5 e4e5": ["c8f5", "g8h6"],  # Advance
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4": ["c8f5", "g8d7"],  # Classical

    # --- Queen's Gambit ---
    "d2d4 d7d5 c2c4": ["e7e6", "c7c6", "d5c4", "e7e5", "g8f6"],
    "d2d4 d7d5 c2c4 e7e6 b1c3": ["g8f6", "f8e7", "c7c5"],
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5": ["f8e7", "h7h6", "c7c5"],  # QGD Classical
    "d2d4 d7d5 c2c4 c7c6 g1f3": ["g8f6", "e7e6", "d5c4"],  # Slav

    # --- King's Indian Defense ---
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4": ["d7d6", "e8g8"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3": ["e8g8"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2": ["e7e5", "c7c5"],

    # --- Nimzo-Indian Defense ---
    "d2d4 g8f6 c2c4 e7e6 b1c3": ["f8b4", "d7d5", "c7c5"],
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2": ["e8g8", "c7c5", "d7d5"],
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3": ["e8g8", "c7c5", "b7b6"],

    # --- Queen's Indian Defense ---
    "d2d4 g8f6 c2c4 e7e6 g1f3": ["b7b6", "f8b4", "d7d5"],
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 b1c3": ["c8b7", "f8b4", "f8e7"],

    # --- Benoni Defense ---
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3": ["e6d5", "d7d6"],
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5": ["d7d6", "g7g6"],

    # --- Grünfeld Defense ---
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5": ["d5c4", "c7c6", "f8g7"],
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4": ["d5c3", "d5f6"],
}

# ---------------------------------------------------------------------------
# Famous historical games organised by opening / theme
# Each entry is a short human-readable description.
# ---------------------------------------------------------------------------

FAMOUS_GAMES: dict[str, list[str]] = {
    "King's Gambit": [
        "Anderssen vs Kieseritzky, 1851 (The Immortal Game): White sacrifices rooks and bishops to deliver "
        "a spectacular checkmate — a defining masterpiece of romantic chess.",
        "Spassky vs Bronstein, 1960 (USSR Championship): Spassky's brilliant King's Gambit crushes the "
        "Soviet champion with sweeping piece activity.",
        "Fischer vs Spassky, 1972 WC Game 3: Fischer plays the King's Gambit (his first public use), "
        "creating an unexpected psychological blow.",
        "Chigorin vs Schiffers, 1897: Attacking brilliancy in the King's Gambit Accepted.",
        "Morphy vs Anderssen, 1858: Dynamic King's Gambit in Morphy's European tour.",
    ],
    "Ruy Lopez": [
        "Fischer vs Spassky, 1972 WC Game 6 (perhaps the greatest game ever played): Fischer dismantles "
        "Spassky's Ruy Lopez with flawless technique and deep preparation.",
        "Kasparov vs Karpov, 1985 WC Game 16 (Kasparov's Immortal): Queen sacrifice on d7 in the Ruy "
        "Lopez to win the World Championship.",
        "Morphy vs Duke of Brunswick, 1858 (Opera Game): Morphy demolishes amateur opposition with rapid "
        "development and a rook sacrifice in the Ruy Lopez.",
        "Lasker vs Capablanca, 1921 WC: Capablanca's perfect technique in the Ruy Lopez endgame.",
        "Karpov vs Korchnoi, 1974 Candidates: Karpov's positional mastery in the Ruy Lopez Closed.",
        "Steinitz vs von Bardeleben, 1895: Famous forced mating sequence from the Ruy Lopez.",
    ],
    "Sicilian Defense": [
        "Fischer vs Tal, 1959 Candidates (Sicilian Najdorf): Fischer's brilliant attacking idea meets "
        "Tal's tactical genius.",
        "Kasparov vs Topalov, 1999 (Kasparov's Immortal, Sicilian Najdorf): King walk and rook sacrifice "
        "produce the most celebrated game of the 20th century.",
        "Deep Blue vs Kasparov, 1997 Game 2: The computer's 36th move Bd6 shocked the chess world.",
        "Fischer vs Larsen, 1971 Candidates (Sicilian): Fischer's demolition of a grandmaster.",
        "Tal vs Smyslov, 1959 (Sicilian): Tal's brilliant queen sacrifice in a wild tactical melee.",
        "Anand vs Kramnik, 2008 WC (Sicilian): Anand's energetic play reclaims the World Championship.",
        "Polugaevsky vs Nezhmetdinov, 1958 (Sicilian): King march in the middlegame — legendary game.",
    ],
    "French Defense": [
        "Tal vs Petrosian, 1958 (French Winawer): Tal's breathtaking piece sacrifice against the "
        "future World Champion.",
        "Nimzowitsch vs Capablanca, 1914 (French): Capablanca's precise positional play.",
        "Botvinnik vs Tal, 1960 WC (French): Tal's dynamic counterplay in the French Defense.",
        "Korchnoi vs Karpov, 1978 WC (French): Epic French Defense in the long World Championship match.",
    ],
    "Caro-Kann Defense": [
        "Tal vs Smyslov, 1959 (Caro-Kann): Tal sacrifices to break open the Caro-Kann.",
        "Capablanca vs Tartakower, 1924 (Caro-Kann): Capablanca's endgame conversion masterclass.",
        "Karpov vs Kasparov, 1986 WC (Caro-Kann): Karpov's positional squeeze.",
        "Anand vs Shirov, 1995 (Caro-Kann): Explosive sacrifices in a Caro-Kann Advance.",
    ],
    "Italian Game": [
        "Greco vs NN, 1619 (Giuoco Piano): One of the earliest recorded brilliant combinations.",
        "Morphy vs Anderssen, 1858: Morphy's flowing development in the Italian Game.",
        "Carlsen vs Caruana, 2014 (Italian Game): Modern treatment of the Giuoco Piano at the top level.",
        "Nakamura vs Carlsen, 2016 (Italian Game): Razor-sharp play in the modern Italian.",
    ],
    "Queen's Gambit": [
        "Lasker vs Capablanca, 1921 WC (QGD): Capablanca's technical masterpiece in the Queen's Gambit "
        "Declined — perfect endgame technique.",
        "Karpov vs Kasparov, 1984 WC Game 9 (QGD Tartakower): Karpov's positional masterpiece.",
        "Petrosian vs Spassky, 1966 WC (QGD): Petrosian's defensive brilliance.",
        "Reshevsky vs Najdorf, 1953 Candidates (QGD): Classic battle in the Queen's Gambit.",
        "Carlsen vs Anand, 2013 WC (QGD): Carlsen's relentless endgame technique.",
        "Botvinnik vs Capablanca, 1938 AVRO (QGD): Botvinnik's famous immortal queen sacrifice.",
    ],
    "King's Indian Defense": [
        "Byrne vs Fischer, 1956 (Game of the Century, KID): 13-year-old Fischer's queen sacrifice stuns "
        "the chess world.",
        "Bronstein vs Boleslavsky, 1950 (KID Classical): Bronstein's creative piece play.",
        "Kasparov vs Karpov, 1993 WC Game 9 (KID): Kasparov's rook sacrifice in the King's Indian.",
        "Tal vs Geller, 1979 (KID): Tal's attacking brilliancy in the Classical KID.",
        "Fischer vs Myagmarsuren, 1967 (KID): Fischer's famous pawn sacrifice leading to a brilliant attack.",
        "Kasparov vs Topalov, 1994 (KID Polugaevsky Variation): Kasparov's deep opening preparation.",
    ],
    "Nimzo-Indian Defense": [
        "Botvinnik vs Capablanca, 1938 AVRO (Nimzo-Indian): Famous queen sacrifice.",
        "Kasparov vs Portisch, 1983 (Nimzo-Indian): Kasparov's brilliant attacking play.",
        "Karpov vs Kasparov, 1986 WC (Nimzo-Indian): Karpov's defensive resourcefulness.",
        "Reshevsky vs Petrosian, 1953 Candidates (Nimzo-Indian): Strategic masterclass.",
    ],
    "Queen's Indian Defense": [
        "Petrosian vs Spassky, 1969 WC (Queen's Indian): Petrosian's prophylactic genius.",
        "Karpov vs Miles, 1986 (Queen's Indian): Positional domination.",
        "Kramnik vs Leko, 2004 WC (Queen's Indian): Kramnik's Berlin-style technique.",
    ],
    "Grünfeld Defense": [
        "Kasparov vs Karpov, 1986 WC (Grünfeld): Kasparov's dynamic play in the Grünfeld.",
        "Fischer vs Portisch, 1970 (Grünfeld): Fischer's perfect technique against the Grünfeld.",
        "Botvinnik vs Reshevsky, 1948 WC (Grünfeld): Classic treatment.",
    ],
    "Benoni Defense": [
        "Tal vs Koblents, 1957 (Benoni): Tal's piece sacrifice in the Benoni.",
        "Fischer vs Petrosian, 1971 Candidates (Benoni): Fischer's relentless pressure.",
        "Kasparov vs Ribli, 1986 (Benoni): Kasparov's kingside attack.",
    ],
    "Dutch Defense": [
        "Nimzowitsch vs Tartakower, 1925 (Dutch): Nimzowitsch's prophylactic masterpiece.",
        "Botvinnik vs Chekhover, 1935 (Dutch Leningrad): Botvinnik's attacking play.",
        "Short vs Timman, 1991 (Dutch): Short's king march on the queenside.",
    ],
    "English Opening": [
        "Karpov vs Kasparov, 1984 WC (English): Karpov's famous squeeze in the English.",
        "Fischer vs Petrosian, 1970 (English): Fischer's dynamic use of the English Opening.",
        "Botvinnik vs Capablanca, 1938 AVRO (English setup): Complex strategic battle.",
    ],
    "Endgame Masterpieces": [
        "Capablanca vs Tartakower, 1924 (Rook endgame): Capablanca's 'ideal' rook endgame technique.",
        "Rubinstein vs Lasker, 1909 (Rook endgame): Rubinstein's immortal rook endgame.",
        "Fischer vs Spassky, 1972 WC Game 6 (Endgame): Fischer's dominating knight vs. bishop endgame.",
        "Karpov vs Kasparov, 1984-85 WC (Endgame technique): Karpov's technical precision.",
        "Réti vs Alekhine, 1925 (Endgame study): Réti's famous theoretical king-and-pawn position.",
        "Lucena Position (historical study): The standard winning rook endgame with passed pawn.",
        "Philidor Position (historical study): The defensive draw in rook vs. rook-and-pawn.",
    ],
    "Tactical Masterpieces": [
        "Topalov vs Shirov, 1998 (Najdorf Sicilian endgame): Shirov's Bh3!! sacrifice — the most "
        "beautiful endgame move ever played.",
        "Rotlewi vs Rubinstein, 1907 (Rubinstein's Immortal): Four pieces sacrificed for checkmate.",
        "Kasparov vs Topalov, 1999: The king march and rook sacrifice.",
        "Morphy vs Allies, 1858 (Opera Game): Four-move combination after rapid development.",
        "Tal vs Vasyukov, 1957: Tal's knight sacrifice opening a kingside attack.",
        "Anderssen vs Dufresne, 1852 (Evergreen Game): Double bishop sacrifice.",
        "Polugaevsky vs Nezhmetdinov, 1958: The king is marched into battle mid-game.",
    ],
}

# ---------------------------------------------------------------------------
# Helper: opening name lookup
# ---------------------------------------------------------------------------


def _moves_key(board: chess.Board) -> str:
    """Return the board's move history as a space-separated UCI string."""
    return " ".join(m.uci() for m in board.move_stack)


def _seq_matches(seq: str, key: str) -> bool:
    """
    Return True when *seq* is a prefix of (or equal to) the move history *key*.

    An empty *seq* represents the starting position and matches any history.
    A non-empty *seq* matches only when the history starts with exactly that
    sequence (followed by a space or equal to it), preventing partial-word
    matches such as "e2e4" inadvertently matching "e2e4x..." variants.
    """
    if not seq:
        return True
    return seq == key or key.startswith(seq + " ")


def get_opening_name(board: chess.Board) -> str | None:
    """
    Return the most specific ECO opening name for the current position,
    or None if the position is not in the database.
    """
    key = _moves_key(board)
    best_name: str | None = None
    best_len = -1

    for seq, (_, name) in ECO_OPENINGS.items():
        seq_len = len(seq.split()) if seq else 0
        if _seq_matches(seq, key) and seq_len > best_len:
            best_len = seq_len
            best_name = name

    return best_name


def get_eco_code(board: chess.Board) -> str | None:
    """
    Return the ECO code for the most specific opening match, or None.
    """
    key = _moves_key(board)
    best_code: str | None = None
    best_len = -1

    for seq, (code, _) in ECO_OPENINGS.items():
        seq_len = len(seq.split()) if seq else 0
        if _seq_matches(seq, key) and seq_len > best_len:
            best_len = seq_len
            best_code = code

    return best_code


# ---------------------------------------------------------------------------
# Helper: opening theory moves
# ---------------------------------------------------------------------------


def get_theoretic_moves(board: chess.Board) -> list[str]:
    """
    Return a list of theoretically recommended moves for the current player.

    Looks up the exact position (identified by the board's move history) in
    the embedded theory database.  Returns moves filtered to the legal move
    list so the caller never receives an illegal suggestion.  Returns an
    empty list if the position is not in the database or it is White to move.
    """
    if board.turn != chess.BLACK:
        return []

    key = _moves_key(board)
    theory_moves = THEORY.get(key)
    if not theory_moves:
        return []

    legal = {m.uci() for m in board.legal_moves}
    return [m for m in theory_moves if m in legal]


# ---------------------------------------------------------------------------
# Helper: historical game lookup
# ---------------------------------------------------------------------------


def get_related_games(opening_name: str | None) -> list[str]:
    """
    Return a list of famous game descriptions related to the given opening name.

    Performs a case-insensitive substring search across all FAMOUS_GAMES keys.
    Always falls back to "Tactical Masterpieces" so callers receive at least
    some historical context.
    """
    if not opening_name:
        return FAMOUS_GAMES.get("Tactical Masterpieces", [])

    name_lower = opening_name.lower()
    results: list[str] = []
    for category, games in FAMOUS_GAMES.items():
        if category.lower() in name_lower or name_lower in category.lower():
            results.extend(games)

    if not results:
        # Broader word-level match
        name_words = set(name_lower.split())
        for category, games in FAMOUS_GAMES.items():
            cat_words = set(category.lower().split())
            if name_words & cat_words:
                results.extend(games)

    return results or FAMOUS_GAMES.get("Tactical Masterpieces", [])


# ---------------------------------------------------------------------------
# Helper: Polyglot opening book
# ---------------------------------------------------------------------------

# Environment variable that can point to a Polyglot .bin file.
_BOOK_ENV_VAR = "CHESS_OPENING_BOOK"

# Candidate paths to search when no explicit path is given.
_DEFAULT_BOOK_PATHS: list[str] = [
    os.environ.get(_BOOK_ENV_VAR, ""),
    os.path.join(os.path.dirname(__file__), "opening_book.bin"),
    os.path.join(os.path.expanduser("~"), ".chess", "books", "opening.bin"),
    "opening_book.bin",
]


def get_book_moves(board: chess.Board, book_path: str | None = None) -> list[str]:
    """
    Return the top moves from a Polyglot binary opening book (.bin file).

    Searches ``book_path`` if provided, otherwise checks the candidate paths
    in ``_DEFAULT_BOOK_PATHS``.  Returns an empty list if no book is found or
    if the position is not in the book.

    Results are sorted by weight (strongest book moves first) and capped at 5.
    """
    paths_to_try: list[str] = [book_path] if book_path else _DEFAULT_BOOK_PATHS

    for path in paths_to_try:
        if not path or not os.path.isfile(path):
            continue
        try:
            with chess.polyglot.open_reader(path) as reader:
                entries = sorted(reader.find_all(board), key=lambda e: -e.weight)
                return [e.move.uci() for e in entries[:5]]
        except Exception:
            continue

    return []
