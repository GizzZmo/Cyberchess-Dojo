"""
Cyberchess Dojo — Main Arena

Runs automated chess games between Stockfish (White) and an LLM-powered
multi-agent orchestrator (Black), recording every game to ``training_data.pgn``
for use as a fine-tuning dataset.

Quick start::

    # Single game with Gemini (default)
    python cyberchess.py

    # Play 10 games in loop mode
    python cyberchess.py --games 10

    # Play with OpenAI GPT-4o
    python cyberchess.py --llm openai --model gpt-4o

    # Play with Anthropic Claude
    python cyberchess.py --llm claude

    # Increase Stockfish difficulty and enable the live dashboard
    python cyberchess.py --skill 10 --games 5 --dashboard

Run ``python cyberchess.py --help`` for the full option list.
"""

import argparse
import datetime
import json
import os
import random

import chess
import chess.engine
import chess.pgn

from orchestrator import ChessOrchestrator

# ---------------------------------------------------------------------------
# Default configuration  (overridden by CLI args or environment variables)
# ---------------------------------------------------------------------------

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "YOUR_STOCKFISH_PATH_HERE")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

STOCKFISH_SKILL_LEVEL = 5       # 0 (weakest) – 20 (Grandmaster)
STOCKFISH_TIME_LIMIT = 0.1      # seconds per move
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Number of independent LLM samples generated per move; the strongest
# candidate is selected via a grandmaster ranking call (best-of-N sampling).
BEST_OF_N = 3

# Files written during a run
_PGN_FILE = "training_data.pgn"
_STATE_FILE = "game_state.json"    # live board state for the web dashboard


# ---------------------------------------------------------------------------
# Legacy single-move helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def get_gemini_move(board: chess.Board, model, retries: int = 3) -> chess.Move:
    """
    Ask a Gemini-style model for the best move on the given board.

    This is the original standalone function retained for backward
    compatibility.  New code should prefer the ``ChessOrchestrator``.

    Args:
        board:   The current chess board.
        model:   Any object with a ``generate_content(prompt)`` method that
                 returns a response with a ``.text`` attribute (e.g. a
                 ``genai.GenerativeModel`` or a ``BaseLLMAdapter``).
        retries: Maximum retry attempts for illegal / malformed moves.

    Returns:
        A legal ``chess.Move``.
    """
    legal_moves = [move.uci() for move in board.legal_moves]

    prompt = f"""
    You are playing a game of Chess against Stockfish.  You are playing Black.

    Current Board Position (FEN): {board.fen()}

    Here is the list of legally possible moves you can make:
    {', '.join(legal_moves)}

    Your goal is to survive and learn.  Analyse the board.
    Pick the best move from the legal list above.

    IMPORTANT: Reply ONLY with the move in UCI format (e.g., e7e5).
    Do not write any other text.
    """

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            move_str = response.text.strip().replace("\n", "").replace(" ", "").replace("`", "")

            if not (4 <= len(move_str) <= 5):
                print(f" > Unexpected move string length: '{move_str}'.  Retrying...")
                prompt += (
                    f"\n\nERROR: '{move_str}' has an unexpected length.  "
                    "Please reply ONLY with a valid UCI move (e.g., e7e5 or e7e8q)."
                )
                continue

            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move

            print(f" > Illegal move: {move_str}.  Retrying...")
            prompt += f"\n\nERROR: {move_str} is not a legal move.  Choose strictly from the provided list."

        except Exception as e:
            print(f" > Error parsing model response: {e}")
            prompt += "\n\nERROR: Invalid format.  Please reply ONLY with the move string (e.g., e7e5)."

    print(" > Model failed to produce a legal move.  Playing random move.")
    return random.choice(list(board.legal_moves))


# ---------------------------------------------------------------------------
# Dashboard state helper
# ---------------------------------------------------------------------------

def _write_game_state(state: dict) -> None:
    """Write current game state to JSON for the web dashboard to poll."""
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(stockfish_path: str, api_key: str, provider: str) -> None:
    """Raise ``ValueError`` if required configuration is missing."""
    if not stockfish_path or stockfish_path.strip() == "YOUR_STOCKFISH_PATH_HERE":
        raise ValueError(
            "STOCKFISH_PATH is not set.  Set the STOCKFISH_PATH environment variable "
            "or pass --stockfish on the command line."
        )
    if provider == "gemini" and (not api_key or api_key.strip() == "YOUR_GEMINI_API_KEY_HERE"):
        raise ValueError(
            "GOOGLE_API_KEY is not set.  Set the GOOGLE_API_KEY environment variable "
            "or pass --api-key on the command line."
        )


# ---------------------------------------------------------------------------
# Single-game runner
# ---------------------------------------------------------------------------

def play_game(
    adapter,
    stockfish_path: str,
    stockfish_skill: int,
    stockfish_time: float,
    best_of_n: int,
    game_number: int = 1,
    live_dashboard: bool = False,
    ai_model_name: str = "AI",
) -> chess.Board:
    """
    Play a single game of Stockfish (White) vs. the AI orchestrator (Black).

    Args:
        adapter:          LLM adapter (any ``BaseLLMAdapter`` or ``GenerativeModel``).
        stockfish_path:   Path to the Stockfish binary.
        stockfish_skill:  Stockfish skill level (0-20).
        stockfish_time:   Seconds Stockfish is allowed per move.
        best_of_n:        Number of LLM samples per move for best-of-N selection.
        game_number:      Sequential game number (used for logging and PGN headers).
        live_dashboard:   If ``True``, write live board state to ``game_state.json``.
        ai_model_name:    Display name for the AI player.

    Returns:
        The completed ``chess.Board``.
    """
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(
            f"Stockfish executable not found at '{stockfish_path}'.  "
            "Set the STOCKFISH_PATH environment variable or use --stockfish."
        )

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise RuntimeError(
            f"Failed to start Stockfish at '{stockfish_path}': {exc}."
        ) from exc

    engine.configure({"Skill Level": stockfish_skill})

    orchestrator = ChessOrchestrator(adapter)
    board = chess.Board()

    print(f"\n{'=' * 60}")
    print(f"  GAME {game_number}: Stockfish Skill {stockfish_skill} (White) vs {ai_model_name} (Black)")
    print(f"{'=' * 60}")

    while not board.is_game_over():
        print(f"\nMove {board.fullmove_number}")
        print(board)

        # Write live state for the dashboard
        if live_dashboard:
            from orchestrator import _piece_count, _OPENING_MOVE_LIMIT, _ENDGAME_PIECE_THRESHOLD
            pc = _piece_count(board)
            phase = (
                "opening" if board.fullmove_number <= _OPENING_MOVE_LIMIT
                else "endgame" if pc <= _ENDGAME_PIECE_THRESHOLD
                else "middlegame"
            )
            _write_game_state({
                "active": True,
                "game_number": game_number,
                "fen": board.fen(),
                "move_number": board.fullmove_number,
                "turn": "White" if board.turn == chess.WHITE else "Black",
                "phase": phase,
                "last_move": board.peek().uci() if board.move_stack else None,
                "stockfish_skill": stockfish_skill,
                "ai_model": ai_model_name,
            })

        if board.turn == chess.WHITE:
            print("Stockfish is thinking...")
            result = engine.play(board, chess.engine.Limit(time=stockfish_time))
            board.push(result.move)
            print(f"Stockfish played: {result.move.uci()}")

        else:
            print("AI Orchestrator is thinking...")
            move = orchestrator.get_best_move(board, n=best_of_n)
            board.push(move)
            print(f"AI played: {move.uci()}")

    print("\n--- GAME OVER ---")
    print(f"Result: {board.result()}")

    engine.quit()

    # Write final (inactive) state to the dashboard
    if live_dashboard:
        _write_game_state({
            "active": False,
            "game_number": game_number,
            "fen": board.fen(),
            "result": board.result(),
            "move_number": board.fullmove_number,
            "stockfish_skill": stockfish_skill,
            "ai_model": ai_model_name,
        })

    return board


# ---------------------------------------------------------------------------
# PGN persistence
# ---------------------------------------------------------------------------

def save_game_data(
    board: chess.Board,
    stockfish_skill: int,
    ai_model_name: str,
    game_number: int = None,
) -> None:
    """
    Append a completed game to the PGN training dataset.

    Args:
        board:           The completed board (contains the full move list).
        stockfish_skill: Stockfish skill level used during the game.
        ai_model_name:   Name / label of the AI player.
        game_number:     Optional sequential game number for the Round header.
    """
    pgn_game = chess.pgn.Game.from_board(board)
    pgn_game.headers["Event"] = "Cyberchess Dojo"
    pgn_game.headers["White"] = f"Stockfish Level {stockfish_skill}"
    pgn_game.headers["Black"] = ai_model_name
    now = datetime.datetime.now()
    pgn_game.headers["Date"] = now.strftime("%Y.%m.%d")
    pgn_game.headers["Time"] = now.strftime("%H:%M:%S")
    if game_number is not None:
        pgn_game.headers["Round"] = str(game_number)

    with open(_PGN_FILE, "a") as f:
        f.write(str(pgn_game) + "\n\n")

    print(f"Game saved to '{_PGN_FILE}'")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cyberchess Dojo — AI Chess Training Arena",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loop mode
    parser.add_argument(
        "--games", type=int, default=1, metavar="N",
        help="Number of games to play in sequence",
    )

    # Stockfish options
    sf_group = parser.add_argument_group("Stockfish")
    sf_group.add_argument(
        "--stockfish", default=STOCKFISH_PATH, metavar="PATH",
        help="Path to the Stockfish binary",
    )
    sf_group.add_argument(
        "--skill", type=int, default=STOCKFISH_SKILL_LEVEL, metavar="0-20",
        help="Stockfish skill level (0 = weakest, 20 = Grandmaster)",
    )
    sf_group.add_argument(
        "--time", type=float, default=STOCKFISH_TIME_LIMIT, metavar="SECS",
        help="Seconds Stockfish is allowed per move",
    )

    # LLM / sampling
    llm_group = parser.add_argument_group("LLM provider")
    llm_group.add_argument(
        "--llm", default="gemini",
        choices=["gemini", "openai", "claude"],
        help="LLM provider to use as the AI student",
    )
    llm_group.add_argument(
        "--model", default=None, metavar="MODEL_NAME",
        help="Model name override (e.g. gpt-4o, claude-3-5-sonnet-20241022)",
    )
    llm_group.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="API key (falls back to the provider environment variable)",
    )
    llm_group.add_argument(
        "--best-of-n", type=int, default=BEST_OF_N, metavar="N",
        help="Number of LLM samples per move for best-of-N selection",
    )

    # Dashboard
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Write live board state to game_state.json for the web dashboard",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve Stockfish path and API key
    stockfish_path = args.stockfish
    api_key = args.api_key
    if args.llm == "gemini" and not api_key:
        api_key = GOOGLE_API_KEY

    _validate_config(stockfish_path, api_key, args.llm)

    # Create the LLM adapter
    from llm_adapter import create_adapter
    adapter = create_adapter(provider=args.llm, model_name=args.model, api_key=api_key)
    ai_label = f"{args.llm.capitalize()} ({adapter.model_name})"

    print(f"LLM provider : {ai_label}")
    print(f"Stockfish    : {stockfish_path}  (skill={args.skill}, time={args.time}s)")
    print(f"Best-of-N    : {args.best_of_n}")
    print(f"Games to play: {args.games}")

    # Elo tracking — load existing history and show current rating
    from elo_tracker import EloTracker
    elo = EloTracker()
    if elo.games_played > 0:
        print(f"\n{elo.report()}\n")

    # Dashboard hint
    if args.dashboard:
        print("\n📊 Dashboard mode enabled — game_state.json will be updated live.")
        print("   Open a second terminal and run:  python dashboard.py")
        print("   Then visit:  http://127.0.0.1:5000\n")

    # ------------------------------------------------------------------ #
    #  Main loop — play N games in sequence (loop mode)                  #
    # ------------------------------------------------------------------ #
    for game_idx in range(1, args.games + 1):
        if args.games > 1:
            print(f"\n{'#' * 60}")
            print(f"  GAME {game_idx} of {args.games}")
            print(f"{'#' * 60}")

        board = play_game(
            adapter=adapter,
            stockfish_path=stockfish_path,
            stockfish_skill=args.skill,
            stockfish_time=args.time,
            best_of_n=args.best_of_n,
            game_number=game_idx,
            live_dashboard=args.dashboard,
            ai_model_name=ai_label,
        )

        save_game_data(board, args.skill, ai_label, game_number=game_idx)

        # Update and display Elo
        prev_elo = elo.current_elo
        delta = elo.update(
            result=board.result(),
            opponent_skill=args.skill,
            game_number=game_idx,
            ai_color="black",
        )
        sign = "+" if delta >= 0 else ""
        print(f"\n📈 Elo update: {prev_elo:.0f} → {elo.current_elo:.0f}  ({sign}{delta:.0f})")

    # ------------------------------------------------------------------ #
    #  Final summary                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{elo.report()}")
    print(f"\nTraining data saved to '{_PGN_FILE}'")
    print("Elo history saved to 'elo_history.json'")
    print("Run `python finetune_pipeline.py` to generate a fine-tuning dataset.")
    print("Run `python dashboard.py` to view the web dashboard.")


if __name__ == "__main__":
    main()
