import chess
import chess.engine
import chess.pgn
import google.generativeai as genai
import os
import random
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

# Number of independent LLM samples generated per move; the strongest
# candidate is selected via a grandmaster ranking call (best-of-N sampling).
BEST_OF_N = 3

# --- STARTUP VALIDATION ---
if (STOCKFISH_PATH == "YOUR_STOCKFISH_PATH_HERE" or not STOCKFISH_PATH.strip()
        or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY.strip()):
    raise ValueError(
        "Please update STOCKFISH_PATH and GOOGLE_API_KEY in the configuration. "
        "Set the STOCKFISH_PATH and GOOGLE_API_KEY environment variables, or edit "
        "the constants directly in this file."
    )

# Setup Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def get_gemini_move(board, retries=3):
    """
    Sends the board state to Gemini and asks for a move.
    Includes a retry loop for illegal moves.
    """
    legal_moves = [move.uci() for move in board.legal_moves]

    # We provide the FEN (Board State) and the list of legal moves to help Gemini
    # ground its reasoning and avoid hallucinations.
    prompt = f"""
    You are playing a game of Chess against Stockfish. You are playing Black.
    
    Current Board Position (FEN): {board.fen()}
    
    Here is the list of legally possible moves you can make:
    {', '.join(legal_moves)}
    
    Your goal is to survive and learn. Analyze the board.
    Pick the best move from the legal list above.
    
    IMPORTANT: Reply ONLY with the move in UCI format (e.g., e7e5). Do not write any other text.
    """

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            move_str = response.text.strip().replace("\n", "").replace(" ", "")

            # clean up common formatting issues if Gemini adds markdown
            move_str = move_str.replace("`", "")

            # UCI moves are 4 chars (e.g. e7e5) or 5 chars for promotions (e.g. e7e8q)
            if not (4 <= len(move_str) <= 5):
                print(f" > Gemini returned a move string with unexpected length: '{move_str}'. Retrying...")
                prompt += (
                    f"\n\nERROR: '{move_str}' has an unexpected length. "
                    "Please reply ONLY with a valid UCI move (e.g., e7e5 or e7e8q)."
                )
                continue

            move = chess.Move.from_uci(move_str)

            if move in board.legal_moves:
                return move
            else:
                print(f" > Gemini tried illegal move: {move_str}. Retrying...")
                # Add feedback to the next prompt (In-Context Learning)
                prompt += f"\n\nERROR: {move_str} is not a legal move. Please choose strictly from the provided list."

        except Exception as e:
            print(f" > Error parsing Gemini response: {e}")
            prompt += f"\n\nERROR: Invalid format. Please reply ONLY with the move string (e.g., e7e5)."

    # If Gemini fails 3 times, we make a random move to keep the game going (fallback)
    print(" > Gemini failed to produce a legal move. Making random move.")
    return random.choice(list(board.legal_moves))


def play_game():
    # Initialize Board and Stockfish
    board = chess.Board()

    if not os.path.exists(STOCKFISH_PATH):
        raise FileNotFoundError(
            f"Stockfish executable not found at '{STOCKFISH_PATH}'. "
            "Please set the STOCKFISH_PATH environment variable or update the "
            "STOCKFISH_PATH constant in this file."
        )

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except (FileNotFoundError, PermissionError, OSError) as e:
        raise RuntimeError(
            f"Failed to start Stockfish at '{STOCKFISH_PATH}': {e}. "
            "Please verify that the STOCKFISH_PATH is correct and the binary is executable."
        ) from e

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
            # Best-of-N sampling: the orchestrator generates BEST_OF_N candidate
            # moves from the phase-appropriate agent(s) and selects the strongest
            # via a grandmaster ranking call, improving overall move quality.
            print("Gemini Orchestrator is thinking...")
            move = orchestrator.get_best_move(board, n=BEST_OF_N)
            board.push(move)
            print(f"Gemini played: {move.uci()}")
            game_moves.append(move)

    # --- GAME OVER ---
    print("\n--- GAME OVER ---")
    print(f"Result: {board.result()}")

    engine.quit()
    return board


def save_game_data(board, game_number=None):
    """
    Saves the game to a PGN file.
    This is the dataset we will use later to FINE TUNE Gemini.
    """
    pgn_game = chess.pgn.Game.from_board(board)
    pgn_game.headers["Event"] = "Cyberchess Dojo"
    pgn_game.headers["White"] = f"Stockfish Level {STOCKFISH_SKILL_LEVEL}"
    pgn_game.headers["Black"] = f"Gemini {GEMINI_MODEL_NAME}"
    now = datetime.datetime.now()
    pgn_game.headers["Date"] = now.strftime("%Y.%m.%d")
    pgn_game.headers["Time"] = now.strftime("%H:%M:%S")
    if game_number is not None:
        pgn_game.headers["Round"] = str(game_number)

    with open("training_data.pgn", "a") as f:
        f.write(str(pgn_game) + "\n\n")
    print("Game saved to 'training_data.pgn'")


if __name__ == "__main__":
    # In a real app, you would loop this: while True: play_game()
    finished_board = play_game()
    save_game_data(finished_board, game_number=1)
