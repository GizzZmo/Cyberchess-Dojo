"""
Web Dashboard for Cyberchess-Dojo.

Provides a live browser-based visualisation of ongoing and completed games,
Elo history, and game statistics.

Run the dashboard as a standalone server::

    python dashboard.py                        # http://127.0.0.1:5000
    python dashboard.py --host 0.0.0.0         # accessible on the local network
    python dashboard.py --port 8080            # custom port

For live board updates during a game, run ``cyberchess.py`` with the
``--dashboard`` flag in a separate terminal.  That flag tells the arena to
write the current board state to ``game_state.json``, which the dashboard
polls every 2 seconds.

API endpoints
-------------
``GET /``              — Renders the HTML dashboard (``templates/index.html``).
``GET /api/state``     — Returns the current live game state as JSON.
``GET /api/elo``       — Returns the full Elo history as JSON.
``GET /api/games``     — Returns a list of completed games parsed from the PGN.
"""

import json
import os
from pathlib import Path

import chess
import chess.pgn
import chess.svg
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# File paths — can be overridden via environment variables.
_PGN_FILE   = os.environ.get("PGN_FILE",   "training_data.pgn")
_ELO_FILE   = os.environ.get("ELO_FILE",   "elo_history.json")
_STATE_FILE = os.environ.get("STATE_FILE", "game_state.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    """Safely read a JSON file; return an empty dict on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _parse_pgn_games() -> list[dict]:
    """Parse completed games from the PGN training file."""
    games: list[dict] = []
    pgn_path = Path(_PGN_FILE)

    if not pgn_path.exists():
        return games

    try:
        with open(pgn_path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                headers   = game.headers
                ply_count = sum(1 for _ in game.mainline_moves())

                # Extract Stockfish skill level from the "White" header
                white = headers.get("White", "")
                skill = None
                if "Level" in white:
                    try:
                        skill = int(white.split("Level")[-1].strip())
                    except ValueError:
                        pass

                games.append({
                    "event":           headers.get("Event",  ""),
                    "white":           white,
                    "black":           headers.get("Black",  ""),
                    "result":          headers.get("Result", "*"),
                    "date":            headers.get("Date",   ""),
                    "round":           headers.get("Round",  ""),
                    "moves":           ply_count,
                    "stockfish_skill": skill,
                })
    except Exception:
        pass

    return games


def _board_svg(fen: str, last_move_uci: str = None) -> str:
    """Render a chess board SVG from a FEN string, optionally highlighting the last move."""
    try:
        board  = chess.Board(fen)
        arrows = []
        if last_move_uci:
            try:
                move = chess.Move.from_uci(last_move_uci)
                arrows = [chess.svg.Arrow(move.from_square, move.to_square, color="#e94560")]
            except Exception:
                pass
        return chess.svg.board(board, arrows=arrows, size=400)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """Return the current live game state (written by cyberchess.py --dashboard)."""
    state = _read_json(_STATE_FILE)
    if not state:
        return jsonify({"active": False})

    fen       = state.get("fen")
    last_move = state.get("last_move")
    state["svg"] = _board_svg(fen, last_move) if fen else ""

    return jsonify(state)


@app.route("/api/elo")
def api_elo():
    """Return the full Elo history for the AI player."""
    data = _read_json(_ELO_FILE)
    return jsonify(data if data else {"current_elo": None, "history": []})


@app.route("/api/games")
def api_games():
    """Return a list of completed games from the PGN training file."""
    return jsonify({"games": _parse_pgn_games()})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Cyberchess Dojo — Web Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",  default="127.0.0.1", help="Interface to bind")
    parser.add_argument("--port",  type=int, default=5000, help="TCP port to bind")
    parser.add_argument("--debug", action="store_true",   help="Enable Flask debug mode")
    return parser


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    print(f"🌐  Cyberchess Dojo Dashboard  →  http://{args.host}:{args.port}")
    print(f"    PGN file   : {_PGN_FILE}")
    print(f"    Elo file   : {_ELO_FILE}")
    print(f"    State file : {_STATE_FILE}")
    print()
    print("    For live updates, run the arena in another terminal:")
    print("      python cyberchess.py --dashboard")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
