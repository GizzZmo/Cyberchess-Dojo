#!/usr/bin/env python3
"""Download curated PGN archives into the local pgn_archive folder."""

from __future__ import annotations

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import urllib.error
import urllib.request
import zipfile


DEFAULT_DATASETS: Dict[str, str] = {
    "top100_players": "https://www.pgnmentor.com/players/Top100.zip",
    "tournaments_2024": "https://www.pgnmentor.com/Year/2024.zip",
    "tournaments_2025": "https://www.pgnmentor.com/Year/2025.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PGN archives (Top 100 players, tournaments 2024/2025) into pgn_archive/"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("pgn_archive"),
        help="Destination folder (default: pgn_archive)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Datasets to fetch (default: all predefined datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist for a dataset.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset keys and exit.",
    )
    parser.add_argument(
        "--add-url",
        action="append",
        nargs=2,
        metavar=("NAME", "URL"),
        default=[],
        help="Add or override a dataset mapping (can be provided multiple times).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout in seconds (default: 60).",
    )
    return parser.parse_args()


def build_catalog(additional: Sequence[Sequence[str]]) -> Dict[str, str]:
    catalog = dict(DEFAULT_DATASETS)
    for name, url in additional:
        catalog[name] = url
    return catalog


def fetch_url(url: str, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def extract_zip(data: bytes, destination: Path) -> List[str]:
    extracted: List[str] = []
    with zipfile.ZipFile(BytesIO(data)) as archive:
        archive.extractall(destination)
        extracted.extend(str(destination / member) for member in archive.namelist())
    return extracted


def save_raw(data: bytes, destination: Path, url: str, dataset: str) -> str:
    filename = Path(url).name or f"{dataset}.pgn"
    output_path = destination / filename
    output_path.write_bytes(data)
    return str(output_path)


def has_existing_data(dataset_dir: Path) -> bool:
    for path in dataset_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_dir():
            return True
        if path.is_file():
            return True
    return False


def download_dataset(
    name: str, url: str, dest_root: Path, force: bool, timeout: int
) -> Tuple[str, str, List[str]]:
    dataset_dir = dest_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not force and has_existing_data(dataset_dir):
        logging.info("Skipping %s (already present, use --force to re-download)", name)
        return name, "skipped", []

    logging.info("Downloading %s from %s", name, url)
    try:
        data = fetch_url(url, timeout=timeout)
    except urllib.error.URLError as exc:
        logging.error("Failed to download %s: %s", name, exc)
        return name, "failed", []

    files: List[str]
    if zipfile.is_zipfile(BytesIO(data)):
        files = extract_zip(data, dataset_dir)
        logging.info("Extracted %d files for %s", len(files), name)
    else:
        saved = save_raw(data, dataset_dir, url, name)
        files = [saved]
        logging.info("Saved %s to %s", name, saved)

    return name, "downloaded", files


def list_datasets(catalog: Dict[str, str]) -> None:
    for key, value in catalog.items():
        print(f"{key}: {value}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    catalog = build_catalog(args.add_url)

    if args.list:
        list_datasets(catalog)
        return 0

    selected: Iterable[str] = args.include or catalog.keys()

    dest_root = args.dest
    dest_root.mkdir(parents=True, exist_ok=True)

    downloaded_any = False
    skipped_any = False
    failed_any = False
    for dataset in selected:
        url = catalog.get(dataset)
        if url is None:
            logging.warning("Dataset key '%s' not found; use --add-url to define it", dataset)
            continue
        _, status, files = download_dataset(dataset, url, dest_root, args.force, args.timeout)
        if status == "downloaded" and files:
            downloaded_any = True
        elif status == "skipped":
            skipped_any = True
        elif status == "failed":
            failed_any = True

    if failed_any:
        if downloaded_any or skipped_any:
            logging.warning("Completed with some failures; see log for details.")
        else:
            logging.warning("No files downloaded. Check dataset keys or URLs.")
        return 1

    if downloaded_any:
        logging.info("Download complete.")
        return 0

    if skipped_any:
        logging.info("Datasets already present; nothing to do.")
        return 0

    logging.warning("No files downloaded. Check dataset keys or URLs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
