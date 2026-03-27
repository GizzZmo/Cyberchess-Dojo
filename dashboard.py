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
``GET /about``         — Renders the About page (``templates/about.html``).
``GET /wiki``          — Renders the Wiki / documentation page (``templates/wiki.html``).
``GET /api/state``     — Returns the current live game state as JSON.
``GET /api/elo``       — Returns the full Elo history as JSON.
``GET /api/games``     — Returns a list of completed games parsed from the PGN.
"""

import io
import json
import os
from pathlib import Path

import chess
import chess.pgn
import chess.svg
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# File paths — can be overridden via environment variables.
_PGN_FILE      = os.environ.get("PGN_FILE",      "training_data.pgn")
_ELO_FILE      = os.environ.get("ELO_FILE",      "elo_history.json")
_STATE_FILE    = os.environ.get("STATE_FILE",    "game_state.json")
_PAUSE_FILE    = os.environ.get("PAUSE_FILE",    "pause_flag.json")
_SETTINGS_FILE = os.environ.get("SETTINGS_FILE", "settings.json")

_SETTINGS_DEFAULTS: dict = {
    "stockfish_skill": 5,
    "stockfish_time": 0.1,
    "llm_provider": "gemini",
    "llm_model": "",
    "best_of_n": 3,
    "gemini_api_key": "",
    "openai_api_key": "",
    "anthropic_api_key": "",
}


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


@app.route("/about")
def about():
    """Serve the About page."""
    return render_template("about.html")


@app.route("/wiki")
def wiki():
    """Serve the Wiki / documentation page."""
    return render_template("wiki.html")


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


@app.route("/api/games/export")
def api_games_export():
    """Download the full PGN training file."""
    pgn_path = Path(_PGN_FILE)
    if not pgn_path.exists():
        return jsonify({"error": "No PGN file found"}), 404
    with open(pgn_path, "rb") as f:
        data = f.read()
    return send_file(
        io.BytesIO(data),
        mimetype="application/x-chess-pgn",
        as_attachment=True,
        download_name="cyberchess_games.pgn",
    )


@app.route("/api/games/import", methods=["POST"])
def api_games_import():
    """Append an uploaded PGN file to the training dataset."""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file provided"}), 400
    uploaded = request.files["file"]
    # Guard against excessively large uploads (limit: 10 MB).
    raw = uploaded.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        return jsonify({"ok": False, "error": "File too large (max 10 MB)"}), 413
    content = raw.decode("utf-8", errors="replace")
    # Validate that the upload contains at least one readable game.
    if chess.pgn.read_game(io.StringIO(content)) is None:
        return jsonify({"ok": False, "error": "No valid PGN games found in upload"}), 400
    try:
        with open(_PGN_FILE, "a") as out:
            out.write("\n" + content.strip() + "\n\n")
        return jsonify({"ok": True})
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/pause", methods=["GET"])
def api_pause_get():
    """Return the current pause state."""
    state = _read_json(_PAUSE_FILE)
    return jsonify({"paused": state.get("paused", False)})


@app.route("/api/pause", methods=["POST"])
def api_pause_post():
    """Toggle (or explicitly set) the pause flag for a running game."""
    body = request.get_json(force=True, silent=True) or {}
    current = _read_json(_PAUSE_FILE).get("paused", False)
    # Allow explicit {"paused": true/false} or toggle when key is absent.
    new_state = body.get("paused", not current)
    try:
        with open(_PAUSE_FILE, "w") as f:
            json.dump({"paused": bool(new_state)}, f)
        return jsonify({"paused": bool(new_state)})
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    """Return current settings (API keys are partially masked)."""
    stored = _read_json(_SETTINGS_FILE)
    merged = dict(_SETTINGS_DEFAULTS)
    merged.update(stored)
    # Partially mask API keys so the browser never receives the full secret.
    for key_field in ("gemini_api_key", "openai_api_key", "anthropic_api_key"):
        raw = merged.get(key_field, "")
        if raw and len(raw) > 4:
            merged[key_field] = "****" + raw[-4:]
        elif raw:
            merged[key_field] = "****"
    return jsonify(merged)


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    """Persist settings sent as JSON.  Masked key values are ignored."""
    body = request.get_json(force=True, silent=True) or {}
    stored = _read_json(_SETTINGS_FILE)
    for k, v in body.items():
        # Skip any value that looks like a masked key (starts with ****).
        if isinstance(v, str) and v.startswith("****"):
            continue
        stored[k] = v
    try:
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(stored, f, indent=2)
        return jsonify({"ok": True})
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


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
