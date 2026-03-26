"""
Fine-tuning Data Pipeline for Cyberchess-Dojo.

Reads completed games from ``training_data.pgn`` and converts them into a
JSONL dataset suitable for supervised fine-tuning (SFT) of any LLM.

Each training example is a ``(prompt, completion)`` pair where:
  - **prompt**     — board FEN + legal moves in the same format the agents use
  - **completion** — the UCI move played in that game position

The JSONL format is compatible with:
  * OpenAI fine-tuning API
  * Google Vertex AI supervised tuning
  * Hugging Face ``datasets`` / ``trl`` SFT trainer

Usage::

    # Default: training_data.pgn → finetune_data.jsonl (Black's moves only)
    python finetune_pipeline.py

    # Custom input/output paths
    python finetune_pipeline.py --input my_games.pgn --output dataset.jsonl

    # Include both colours
    python finetune_pipeline.py --all-moves

    # Print stats without writing output
    python finetune_pipeline.py --stats

    # Include per-position metadata (game #, move #, FEN)
    python finetune_pipeline.py --metadata
"""

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.pgn


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------

def pgn_to_training_examples(
    pgn_path: str,
    black_only: bool = True,
) -> list[dict]:
    """
    Parse a PGN file and generate prompt-completion training examples.

    Args:
        pgn_path:   Path to the PGN file.
        black_only: If ``True`` (default), only include moves played by Black
                    (the AI student).  Set to ``False`` to include both colours.

    Returns:
        A list of dicts with ``"prompt"``, ``"completion"``, and ``"metadata"`` keys.
    """
    examples: list[dict] = []

    with open(pgn_path) as f:
        game_index = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_index += 1

            board = game.board()
            for node in game.mainline():
                move = node.move
                color = board.turn  # chess.WHITE or chess.BLACK

                # Optionally skip White's moves
                if black_only and color == chess.WHITE:
                    board.push(move)
                    continue

                legal_moves = [m.uci() for m in board.legal_moves]
                color_str = "Black" if color == chess.BLACK else "White"

                prompt = (
                    f"You are a chess expert playing as {color_str}.\n"
                    f"Current board position (FEN): {board.fen()}\n"
                    f"Legal moves: {', '.join(legal_moves)}\n\n"
                    f"Choose the best move from the legal list above.\n"
                    f"Reply ONLY with the UCI move on the last line (e.g. e7e5)."
                )
                completion = move.uci()

                examples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "metadata": {
                        "game": game_index,
                        "move_number": board.fullmove_number,
                        "color": color_str.lower(),
                        "fen": board.fen(),
                    },
                })

                board.push(move)

    return examples


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_jsonl(
    examples: list[dict],
    output_path: str,
    include_metadata: bool = False,
) -> None:
    """Write training examples to a JSONL file (one JSON object per line)."""
    with open(output_path, "w") as f:
        for ex in examples:
            record: dict = {"prompt": ex["prompt"], "completion": ex["completion"]}
            if include_metadata:
                record["metadata"] = ex["metadata"]
            f.write(json.dumps(record) + "\n")


def print_stats(examples: list[dict]) -> None:
    """Print a summary of the generated dataset."""
    if not examples:
        print("No examples found.")
        return

    games = {ex["metadata"]["game"] for ex in examples}
    colors = [ex["metadata"]["color"] for ex in examples]
    move_nums = [ex["metadata"]["move_number"] for ex in examples]

    print("Dataset Statistics")
    print(f"  Total examples  : {len(examples)}")
    print(f"  Games parsed    : {len(games)}")
    print(f"  Black moves     : {colors.count('black')}")
    print(f"  White moves     : {colors.count('white')}")
    print(f"  Move number min : {min(move_nums)}")
    print(f"  Move number max : {max(move_nums)}")
    avg_ply = sum(move_nums) / len(move_nums)
    print(f"  Avg move number : {avg_ply:.1f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert training_data.pgn to a fine-tuning JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="training_data.pgn",
        help="Input PGN file",
    )
    parser.add_argument(
        "--output", default="finetune_data.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--black-only", dest="black_only", action="store_true", default=True,
        help="Only include Black's (AI student) moves",
    )
    parser.add_argument(
        "--all-moves", dest="black_only", action="store_false",
        help="Include moves for both colours",
    )
    parser.add_argument(
        "--metadata", action="store_true",
        help="Include per-position metadata (game #, move #, FEN) in the output",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print dataset statistics and exit without writing output",
    )
    return parser


def main(argv=None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pgn_path = Path(args.input)
    if not pgn_path.exists():
        print(f"Error: PGN file not found: {pgn_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading games from '{pgn_path}'...")
    examples = pgn_to_training_examples(str(pgn_path), black_only=args.black_only)

    print_stats(examples)

    if args.stats:
        return

    if not examples:
        print("No training examples generated. Exiting.", file=sys.stderr)
        sys.exit(1)

    write_jsonl(examples, args.output, include_metadata=args.metadata)
    print(f"\nWrote {len(examples)} training examples to '{args.output}'")
    print("Fine-tuning format: each line is a JSON object with 'prompt' and 'completion' keys.")
    print("Compatible with: OpenAI fine-tuning API, Vertex AI supervised tuning, Hugging Face datasets.")


if __name__ == "__main__":
    main()
