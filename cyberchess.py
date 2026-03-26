import chess
import chess.engine
import chess.pgn
import google.generativeai as genai
import os
import datetime
from orchestrator import ChessOrchestrator

# --- CONFIGURATION ---
# Set via environment variables, or replace the fallback strings directly.
# Windows example: "C:/Users/Jon/Downloads/stockfish/stockfish-windows-x86-64.exe"
# Mac example: "/opt/homebrew/bin/stockfish"
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "YOUR_STOCKFISH_PATH_HERE")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# Stockfish tuning knobs
STOCKFISH_SKILL_LEVEL = 5   # 0 (weakest) – 20 (Grandmaster)
STOCKFISH_TIME_LIMIT = 0.1  # seconds per move

GEMINI_MODEL_NAME = "gemini-1.5-flash"  # Using Flash for speed

# Setup Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def play_game():
    # Initialize Board and Stockfish
    board = chess.Board()

    if not os.path.exists(STOCKFISH_PATH):
        raise FileNotFoundError(
            f"Stockfish executable not found at '{STOCKFISH_PATH}'. "
            "Please set the STOCKFISH_PATH environment variable or update the "
            "STOCKFISH_PATH constant in this file."
        )

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Set Stockfish skill level (lower it initially so Gemini has a chance)
    engine.configure({"Skill Level": STOCKFISH_SKILL_LEVEL})

    # Instantiate the AI orchestrator here so agents are only created when a
    # game is actually started (avoids unnecessary overhead on import).
    orchestrator = ChessOrchestrator(model)

    print("--- CYBERCHESS: Stockfish (White) vs Gemini Orchestrator (Black) ---")

    game_moves = []

    while not board.is_game_over():
        print(f"\nMove {board.fullmove_number}")
        print(board)

        if board.turn == chess.WHITE:
            # --- STOCKFISH TURN ---
            print("Stockfish is thinking...")
            # Limit Stockfish to STOCKFISH_TIME_LIMIT seconds so it plays fast
            result = engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
            board.push(result.move)
            print(f"Stockfish played: {result.move.uci()}")
            game_moves.append(result.move)

        else:
            # --- GEMINI ORCHESTRATOR TURN ---
            # The orchestrator detects the game phase, selects the best agent(s),
            # and synthesises a final move when agents disagree.
            print("Gemini Orchestrator is thinking...")
            move = orchestrator.get_move(board)
            board.push(move)
            print(f"Gemini played: {move.uci()}")
            game_moves.append(move)

    # --- GAME OVER ---
    print("\n--- GAME OVER ---")
    print(f"Result: {board.result()}")

    engine.quit()
    return board


def save_game_data(board):
    """
    Saves the game to a PGN file.
    This is the dataset we will use later to FINE TUNE Gemini.
    """
    pgn_game = chess.pgn.Game.from_board(board)
    pgn_game.headers["Event"] = "Cyberchess Dojo"
    pgn_game.headers["White"] = "Stockfish Level 5"
    pgn_game.headers["Black"] = "Gemini Orchestrator (1.5 Flash)"
    pgn_game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")

    with open("training_data.pgn", "a") as f:
        f.write(str(pgn_game) + "\n\n")
    print("Game saved to 'training_data.pgn'")


if __name__ == "__main__":
    # In a real app, you would loop this: while True: play_game()
    finished_board = play_game()
    save_game_data(finished_board)
