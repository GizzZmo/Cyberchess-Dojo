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
import sys
import time

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
MIN_ANALYSIS_TIME = 0.05
MAX_ANALYSIS_TIME = 0.5
MATE_SCORE_THRESHOLD = 100000
ADVANTAGE_THRESHOLD_CP = 50

# Number of independent LLM samples generated per move; the strongest
# candidate is selected via a grandmaster ranking call (best-of-N sampling).
BEST_OF_N = 3

# Files written during a run
_PGN_FILE = "training_data.pgn"
_STATE_FILE = "game_state.json"    # live board state for the web dashboard
_PAUSE_FILE = "pause_flag.json"    # written by dashboard to pause/resume a game
_SETTINGS_FILE = "settings.json"   # persisted settings (written by dashboard)

# Time controls available for training sessions.
TIME_CONTROLS: dict[str, float] = {
    "classic": 0.3,
    "rapid": 0.1,
    "lightning": 0.03,
}

MATCHUPS: dict[str, tuple[str, str]] = {
    "stockfish-ai": ("stockfish", "ai"),
    "stockfish-stockfish": ("stockfish", "stockfish"),
    "ai-ai": ("ai", "ai"),
    "ai-stockfish": ("ai", "stockfish"),
}


# ---------------------------------------------------------------------------
# Pause support
# ---------------------------------------------------------------------------

def _is_paused() -> bool:
    """Return True if the dashboard has requested a pause."""
    try:
        with open(_PAUSE_FILE) as f:
            return json.load(f).get("paused", False)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _wait_if_paused(live_dashboard: bool) -> None:
    """Block execution while a pause is active, printing a status line once."""
    if not live_dashboard or not _is_paused():
        return
    print("  ⏸  Game paused via dashboard — waiting to resume…")
    while _is_paused():
        time.sleep(1)
    print("  ▶  Resumed.")


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


def _stockfish_insights(engine: chess.engine.SimpleEngine, board: chess.Board, analysis_time: float) -> dict:
    """
    Return a lightweight Stockfish evaluation plus top 3 moves for both colors.

    The analysis is intentionally shallow and time-limited so it can be invoked
    on every ply without slowing the game loop.
    """
    insights = {
        "evaluation": None,
        "best_moves": {"white": [], "black": []},
    }

    clamped_time = min(max(analysis_time, MIN_ANALYSIS_TIME), MAX_ANALYSIS_TIME)
    limit = chess.engine.Limit(time=clamped_time)

    def _extract_moves(analysis_board: chess.Board, lines) -> list[dict]:
        if isinstance(lines, dict):
            lines = [lines]
        best: list[dict] = []
        for entry in lines:
            pv = entry.get("pv")
            if not pv:
                continue
            move = pv[0]
            if move not in analysis_board.legal_moves:
                continue
            # History is not required for these display moves; omit the stack
            # to avoid unnecessary copying.
            temp_board = analysis_board.copy(stack=False)
            try:
                san = temp_board.san(move)
            except (ValueError, chess.IllegalMoveError):
                san = move.uci()
            best.append({"uci": move.uci(), "san": san})
        return best

    # Evaluation and best lines for the actual side to move.
    try:
        primary = engine.analyse(board, limit, multipv=3)
        primary_lines = primary if isinstance(primary, list) else [primary]
        if primary_lines:
            score = primary_lines[0].get("score")
            if score:
                pov = score.pov(chess.WHITE)
                cp = pov.score(mate_score=MATE_SCORE_THRESHOLD)
                mate = pov.mate()
                leader = "equal"
                if cp is not None:
                    if cp > ADVANTAGE_THRESHOLD_CP:
                        leader = "white"
                    elif cp < -ADVANTAGE_THRESHOLD_CP:
                        leader = "black"
                insights["evaluation"] = {"cp": cp, "mate": mate, "leader": leader}
            turn_key = "white" if board.turn == chess.WHITE else "black"
            insights["best_moves"][turn_key] = _extract_moves(board, primary_lines)
    except Exception as exc:  # pragma: no cover - runtime logging for diagnostics
        print(f"[dashboard] Stockfish insight error (turn side): {exc}", file=sys.stderr)

    # Hypothetical best lines for the opposite color (null move to switch turn when safe).
    try:
        if not board.is_check():
            opp_board = board.copy(stack=False)
            opp_board.push(chess.Move.null())
            opposite = engine.analyse(opp_board, limit, multipv=3)
            opp_key = "black" if board.turn == chess.WHITE else "white"
            insights["best_moves"][opp_key] = _extract_moves(opp_board, opposite)
    except Exception as exc:  # pragma: no cover - runtime logging for diagnostics
        print(f"[dashboard] Stockfish insight error (opposite side): {exc}", file=sys.stderr)

    return insights


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


def _resolve_matchup(matchup: str) -> tuple[str, str]:
    """Return (white_type, black_type) for the configured matchup."""
    return MATCHUPS[matchup]


def _ai_perspective_color(white_type: str, black_type: str) -> str | None:
    """Return the single AI side color when exactly one side is AI, else None."""
    if white_type == "ai" and black_type == "stockfish":
        return "white"
    if white_type == "stockfish" and black_type == "ai":
        return "black"
    return None


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
    white_type: str = "stockfish",
    black_type: str = "ai",
    time_control_mode: str = "rapid",
) -> tuple[chess.Board, str, str]:
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

    orchestrator = ChessOrchestrator(adapter) if (white_type == "ai" or black_type == "ai") else None
    board = chess.Board()
    white_label = f"Stockfish Level {stockfish_skill}" if white_type == "stockfish" else ai_model_name
    black_label = f"Stockfish Level {stockfish_skill}" if black_type == "stockfish" else ai_model_name

    # Maintain a SAN move list for the dashboard move-list panel.
    san_moves: list[str] = []

    print(f"\n{'=' * 60}")
    print(f"  GAME {game_number}: {white_label} (White) vs {black_label} (Black)")
    print(f"{'=' * 60}")

    while not board.is_game_over():
        # Honour a pause requested from the dashboard before processing each move.
        _wait_if_paused(live_dashboard)

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
            insights = _stockfish_insights(engine, board, stockfish_time)
            _write_game_state({
                "active": True,
                "paused": _is_paused(),
                "game_number": game_number,
                "fen": board.fen(),
                "move_number": board.fullmove_number,
                "turn": "White" if board.turn == chess.WHITE else "Black",
                "phase": phase,
                "last_move": board.peek().uci() if board.move_stack else None,
                "stockfish_skill": stockfish_skill,
                "time_control_mode": time_control_mode,
                "ai_model": ai_model_name,
                "white_player": white_label,
                "black_player": black_label,
                "san_moves": san_moves,
                "evaluation": insights.get("evaluation"),
                "best_moves": insights.get("best_moves"),
            })

        current_side = white_type if board.turn == chess.WHITE else black_type
        if current_side == "stockfish":
            print("Stockfish is thinking...")
            result = engine.play(board, chess.engine.Limit(time=stockfish_time))
            san = board.san(result.move)
            board.push(result.move)
            san_moves.append(san)
            print(f"Stockfish played: {result.move.uci()} ({san})")

        else:
            print("AI Orchestrator is thinking...")
            move = orchestrator.get_best_move(board, n=best_of_n)  # type: ignore[union-attr]
            san = board.san(move)
            board.push(move)
            san_moves.append(san)
            print(f"AI played: {move.uci()} ({san})")

    print("\n--- GAME OVER ---")
    print(f"Result: {board.result()}")

    engine.quit()

    # Write final (inactive) state to the dashboard
    if live_dashboard:
        insights = _stockfish_insights(engine, board, stockfish_time)
        _write_game_state({
            "active": False,
            "paused": False,
            "game_number": game_number,
            "fen": board.fen(),
            "result": board.result(),
            "move_number": board.fullmove_number,
            "stockfish_skill": stockfish_skill,
            "time_control_mode": time_control_mode,
            "ai_model": ai_model_name,
            "white_player": white_label,
            "black_player": black_label,
            "san_moves": san_moves,
            "evaluation": insights.get("evaluation"),
            "best_moves": insights.get("best_moves"),
        })

    return board, white_label, black_label


# ---------------------------------------------------------------------------
# PGN persistence
# ---------------------------------------------------------------------------

def save_game_data(
    board: chess.Board,
    white_label: str,
    black_label: str,
    game_number: int = None,
) -> None:
    """
    Append a completed game to the PGN training dataset.

    Args:
        board:           The completed board (contains the full move list).
        white_label:     Label/name for White player.
        black_label:     Label/name for Black player.
        game_number:     Optional sequential game number for the Round header.
    """
    pgn_game = chess.pgn.Game.from_board(board)
    pgn_game.headers["Event"] = "Cyberchess Dojo"
    pgn_game.headers["White"] = white_label
    pgn_game.headers["Black"] = black_label
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
    sf_group.add_argument(
        "--time-control", default=None, choices=list(TIME_CONTROLS.keys()),
        help="Time-control preset: classic, rapid, or lightning",
    )
    sf_group.add_argument(
        "--matchup", default="stockfish-ai", choices=list(MATCHUPS.keys()),
        help="Player types by color: stockfish-ai, stockfish-stockfish, ai-ai, ai-stockfish",
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
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)

    # Resolve Stockfish path and API key
    stockfish_path = args.stockfish
    api_key = args.api_key
    if args.llm == "gemini" and not api_key:
        api_key = GOOGLE_API_KEY

    white_type, black_type = _resolve_matchup(args.matchup)
    ai_color = _ai_perspective_color(white_type, black_type)
    needs_ai = white_type == "ai" or black_type == "ai"

    if not stockfish_path or stockfish_path.strip() == "YOUR_STOCKFISH_PATH_HERE":
        raise ValueError(
            "STOCKFISH_PATH is not set.  Set the STOCKFISH_PATH environment variable "
            "or pass --stockfish on the command line."
        )
    if needs_ai:
        _validate_config(stockfish_path, api_key, args.llm)

    # Create the LLM adapter
    adapter = None
    ai_label = "AI"
    if needs_ai:
        from llm_adapter import create_adapter
        adapter = create_adapter(provider=args.llm, model_name=args.model, api_key=api_key)
        ai_label = f"{args.llm.capitalize()} ({adapter.model_name})"

    print(f"LLM provider : {ai_label if needs_ai else 'N/A (Stockfish-only matchup)'}")
    # Resolve optional settings from dashboard persistence.
    settings = {}
    try:
        with open(_SETTINGS_FILE) as f:
            settings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        settings = {}

    selected_mode = (args.time_control or settings.get("time_control_mode") or "rapid").lower()
    if selected_mode not in TIME_CONTROLS:
        selected_mode = "rapid"

    # CLI --time has priority; otherwise use selected mode preset.
    explicit_time_flag = "--time" in argv_list
    effective_time = args.time if explicit_time_flag else TIME_CONTROLS[selected_mode]

    print(f"Stockfish    : {stockfish_path}  (skill={args.skill}, time={effective_time}s, mode={selected_mode})")
    print(f"Best-of-N    : {args.best_of_n}")
    print(f"Games to play: {args.games}")
    print(f"Matchup      : {args.matchup}  (White={white_type}, Black={black_type})")

    # Elo tracking — load existing history and show current rating
    from elo_tracker import EloTracker
    from adaptive_system import AdaptiveTrainingManager
    elo = EloTracker()
    adaptive = AdaptiveTrainingManager()
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

        if ai_color:
            plan = adaptive.plan_next_game(
                base_skill=args.skill,
                base_time=effective_time,
                base_best_of_n=args.best_of_n,
                elo_history=elo.history,
            )
        else:
            plan = {
                "stockfish_skill": args.skill,
                "stockfish_time": effective_time,
                "best_of_n": args.best_of_n,
                "recent_score": 0.5,
                "regime": "fixed",
            }

        print(
            f"🧭 Adaptive plan: regime={plan['regime']} | "
            f"recent_score={plan['recent_score']:.2f} | "
            f"skill={plan['stockfish_skill']} | time={plan['stockfish_time']}s | "
            f"best_of_n={plan['best_of_n']}"
        )

        board, white_label, black_label = play_game(
            adapter=adapter,
            stockfish_path=stockfish_path,
            stockfish_skill=plan["stockfish_skill"],
            stockfish_time=plan["stockfish_time"],
            best_of_n=plan["best_of_n"],
            game_number=game_idx,
            live_dashboard=args.dashboard,
            ai_model_name=ai_label,
            white_type=white_type,
            black_type=black_type,
            time_control_mode=selected_mode,
        )

        save_game_data(board, white_label, black_label, game_number=game_idx)

        # Update and display Elo only when exactly one AI plays against Stockfish.
        if ai_color:
            prev_elo = elo.current_elo
            delta = elo.update(
                result=board.result(),
                opponent_skill=plan["stockfish_skill"],
                game_number=game_idx,
                ai_color=ai_color,
            )
            sign = "+" if delta >= 0 else ""
            print(f"\n📈 Elo update: {prev_elo:.0f} → {elo.current_elo:.0f}  ({sign}{delta:.0f})")
        else:
            print("\n📈 Elo update skipped for this matchup (no single AI vs Stockfish perspective).")

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
