## PGN Archive

This folder is reserved for downloaded PGN datasets. Everything here is ignored by Git so you can store large archives locally without committing them.

### Default datasets
- **top100_players** — Top 100 players collection (PGN Mentor).
- **tournaments_2024** — Tournament PGNs for 2024 (PGN Mentor).
- **tournaments_2025** — Tournament PGNs for 2025 (PGN Mentor).

### Download
```bash
python scripts/download_pgn_archive.py
```

Options:
- `--include DATASET [DATASET...]` to limit which datasets to fetch (defaults to all above).
- `--dest PATH` to change the download location (default: `pgn_archive`).
- `--force` to re-download even if files already exist.
- `--list` to print available dataset keys and exit.
- `--add-url NAME URL` to supply additional/custom datasets.

> The script pulls from PGN Mentor’s public archives. If any default URL changes, re-run with `--add-url NAME URL` to point to the updated link.
