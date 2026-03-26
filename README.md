# ♟️ Cyberchess-Dojo

[![CI](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/ci.yml/badge.svg)](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/GizzZmo/Cyberchess-Dojo)](https://github.com/GizzZmo/Cyberchess-Dojo/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **An AI Training Arena** — Stockfish (The Teacher) plays chess against a multi-agent LLM system (The Student), generating PGN training data for future fine-tuning.

---

## 🧠 Concept

Cyberchess-Dojo is an **automated chess training pipeline** where a classical engine and a large language model compete against each other:

| Role | Engine | Colour |
|------|--------|--------|
| 🎓 Teacher | [Stockfish](https://stockfishchess.org/) | White |
| 🤖 Student | LLM (Gemini / GPT-4o / Claude) via AI Orchestrator | Black |

Every game is saved as a [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation) file (`training_data.pgn`). The long-term goal is to use this dataset to **fine-tune the LLM** so it learns from Stockfish's play.

```
┌──────────────────────────────────────────────────────┐
│                   Cyberchess Arena                   │
│                                                      │
│  Stockfish ──(UCI)──► chess.engine                   │
│                            │                         │
│                     Board State (FEN)                │
│                            │                         │
│                    ChessOrchestrator                 │
│                   ┌────────┴────────┐                │
│            phase detection      agent selection      │
│                   │                                  │
│      ┌────────────┼────────────────┐                 │
│  OpeningAgent  TacticalAgent  PositionalAgent        │
│                EndgameAgent                          │
│                   │                                  │
│            LLM Adapter (Gemini / OpenAI / Claude)    │
│                   │                                  │
│            UCI move ──► board.push()                 │
│                   │                                  │
│        training_data.pgn  elo_history.json           │
└──────────────────────────────────────────────────────┘
```

---

## 🤖 AI Agents & Orchestrator

### Specialised Agents (`agents/`)

Each agent is a focused LLM persona with a domain-specific prompt:

| Agent | File | Expertise |
|-------|------|-----------|
| **OpeningAgent** | `agents/opening_agent.py` | Development, centre control, castling |
| **TacticalAgent** | `agents/tactical_agent.py` | Checks, captures, forks, pins, skewers |
| **PositionalAgent** | `agents/positional_agent.py` | Pawn structure, piece activity, weak squares |
| **EndgameAgent** | `agents/endgame_agent.py` | King activity, pawn promotion, technique |

All agents share a common base class (`agents/base_agent.py`) that handles retry logic, UCI move extraction via regex, and a random-move fallback.

### AI Orchestrator (`orchestrator.py`)

The `ChessOrchestrator` coordinates the agents using the following routing logic:

| Game Phase | Condition | Agents Used |
|------------|-----------|-------------|
| **Opening** | Moves 1–10 | OpeningAgent only |
| **Endgame** | ≤ 6 non-pawn pieces remain | EndgameAgent only |
| **Tactical middlegame** | Any check available | TacticalAgent → PositionalAgent |
| **Quiet middlegame** | No checks available | PositionalAgent → TacticalAgent |

When two agents disagree, the orchestrator makes a third call — acting as a grandmaster arbitrator — to synthesise a final decision.

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | |
| [Stockfish](https://stockfishchess.org/download/) | ≥ 15 | Must be installed separately |
| LLM API key | — | See [LLM Provider Setup](#-llm-provider-setup) |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/GizzZmo/Cyberchess-Dojo.git
cd Cyberchess-Dojo
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

**Linux / macOS**
```bash
export STOCKFISH_PATH="/usr/local/bin/stockfish"
export GOOGLE_API_KEY="your-gemini-api-key"
```

**Windows (PowerShell)**
```powershell
$env:STOCKFISH_PATH = "C:\Users\Jon\Downloads\stockfish\stockfish-windows-x86-64.exe"
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

### 4. Run the arena

```bash
# Single game (default)
python cyberchess.py

# Play 10 games in a row (loop mode)
python cyberchess.py --games 10

# Show all options
python cyberchess.py --help
```

---

## ⚙️ Configuration & CLI Reference

All settings can be passed as command-line arguments or set via environment variables.

```
usage: cyberchess.py [-h] [--games N] [--dashboard]
                     [--stockfish PATH] [--skill 0-20] [--time SECS]
                     [--llm {gemini,openai,claude}] [--model MODEL_NAME]
                     [--api-key KEY] [--best-of-n N]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--games N` | `1` | Number of games to play in sequence (loop mode) |
| `--skill 0-20` | `5` | Stockfish strength 0 (weakest) – 20 (Grandmaster) |
| `--time SECS` | `0.1` | Seconds Stockfish spends per move |
| `--llm` | `gemini` | LLM provider: `gemini`, `openai`, or `claude` |
| `--model` | *(provider default)* | Model name override (e.g. `gpt-4o`) |
| `--api-key` | *(env var)* | API key (overrides environment variable) |
| `--best-of-n N` | `3` | LLM samples per move for best-of-N selection |
| `--stockfish PATH` | `$STOCKFISH_PATH` | Path to the Stockfish binary |
| `--dashboard` | off | Write live board state for the web dashboard |

---

## 🔁 Loop Mode

Play multiple games in sequence automatically.  Elo is updated after each game.

```bash
# Play 20 games and track Elo progression
python cyberchess.py --games 20 --skill 5

# Ramp up difficulty — 10 games at skill 10
python cyberchess.py --games 10 --skill 10
```

Each game is appended to `training_data.pgn` with a `Round` header for easy filtering.

---

## 📈 Elo Tracking

The arena estimates the AI's Elo rating after every game using the standard FIDE formula:

- **Stockfish skill → Elo** mapping based on community benchmarks (Skill 0 ≈ 800, Skill 5 ≈ 1500, Skill 20 ≈ 3200).
- **K-factor**: 32 (developing player).
- Results are persisted to `elo_history.json` so ratings carry over between runs.

After each game the terminal shows:

```
📈 Elo update: 1200 → 1185  (-15)

── Elo Rating: 1185 ──
   Games: 3  |  W: 0  D: 1  L: 2
   Game   1: 1-0       | vs Stockfish Skill 5 (≈1500) | Elo 1200 → 1178 (-22)
   Game   2: 1/2-1/2   | vs Stockfish Skill 5 (≈1500) | Elo 1178 → 1193 (+16)
   Game   3: 1-0       | vs Stockfish Skill 5 (≈1500) | Elo 1193 → 1185 (-8)
```

---

## 🤖 LLM Provider Setup

### Google Gemini (default)

```bash
export GOOGLE_API_KEY="your-key"          # https://aistudio.google.com/app/apikey
python cyberchess.py                       # uses gemini-1.5-flash by default
python cyberchess.py --model gemini-1.5-pro
```

### OpenAI GPT-4o

```bash
pip install openai
export OPENAI_API_KEY="your-key"
python cyberchess.py --llm openai
python cyberchess.py --llm openai --model gpt-4o-mini
```

### Anthropic Claude

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"
python cyberchess.py --llm claude
python cyberchess.py --llm claude --model claude-3-haiku-20240307
```

All three providers expose the same interface through `llm_adapter.py`, so the agents and orchestrator work identically regardless of which LLM is selected.

---

## 🌐 Web Dashboard

The dashboard provides a live browser view of the board, Elo history chart, and game log.

**Step 1** — Start the arena with `--dashboard`:
```bash
python cyberchess.py --games 5 --dashboard
```

**Step 2** — In a separate terminal, start the dashboard server:
```bash
python dashboard.py
```

**Step 3** — Open **http://127.0.0.1:5000** in your browser.

The board updates automatically every 2 seconds.  Custom host/port:
```bash
python dashboard.py --host 0.0.0.0 --port 8080
```

**Dashboard API endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live dashboard HTML |
| `GET /api/state` | Current board state (FEN, phase, last move) |
| `GET /api/elo` | Full Elo history JSON |
| `GET /api/games` | Completed games parsed from `training_data.pgn` |

---

## 🧪 Fine-tuning Pipeline

Convert the accumulated PGN games into a JSONL fine-tuning dataset:

```bash
# Default: training_data.pgn → finetune_data.jsonl (Black's moves only)
python finetune_pipeline.py

# Include both colours
python finetune_pipeline.py --all-moves

# Custom paths
python finetune_pipeline.py --input my_games.pgn --output dataset.jsonl

# Print statistics only (no file written)
python finetune_pipeline.py --stats

# Include per-position metadata in the output
python finetune_pipeline.py --metadata
```

Each line of the output is a JSON object with `"prompt"` and `"completion"` keys:

```json
{
  "prompt": "You are a chess expert playing as Black.\nCurrent board position (FEN): ...\nLegal moves: e7e5, d7d5, ...\n\nChoose the best move ...",
  "completion": "e7e5"
}
```

The format is compatible with:
- **OpenAI** fine-tuning API (`openai.fine_tuning.jobs.create`)
- **Google Vertex AI** supervised tuning
- **Hugging Face** `datasets` / `trl` SFT trainer

---

## 🗂️ Project Structure

```
Cyberchess-Dojo/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── pull_request_template.md
│   └── workflows/
│       └── ci.yml              # Lint + syntax check on every push/PR
├── agents/
│   ├── __init__.py             # Package exports
│   ├── base_agent.py           # Shared base class (retry, UCI extraction, fallback)
│   ├── opening_agent.py        # Opening principles specialist
│   ├── tactical_agent.py       # Tactics specialist (checks, captures, forks)
│   ├── positional_agent.py     # Positional / strategic specialist
│   └── endgame_agent.py        # Endgame technique specialist
├── templates/
│   └── index.html              # Web dashboard HTML (served by dashboard.py)
├── orchestrator.py             # ChessOrchestrator — routes board states to agents
├── cyberchess.py               # Main arena script (loop mode, Elo, dashboard)
├── llm_adapter.py              # Unified LLM interface (Gemini, OpenAI, Claude)
├── elo_tracker.py              # Elo rating system with JSON persistence
├── finetune_pipeline.py        # PGN → JSONL fine-tuning dataset generator
├── dashboard.py                # Flask web dashboard server
├── requirements.txt            # Python dependencies
├── training_data.pgn           # Generated — game records for fine-tuning
├── elo_history.json            # Generated — Elo rating history
├── game_state.json             # Generated — live board state for dashboard
├── finetune_data.jsonl         # Generated — fine-tuning dataset
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 🛠️ CI / Workflow

The [CI workflow](.github/workflows/ci.yml) runs on every push and pull request to `main`/`master`:

1. **Install** — `pip install -r requirements.txt` + `flake8`
2. **Syntax errors** — `flake8 --select=E9,F63,F7,F82` (hard fail)
3. **Style warnings** — `flake8 --exit-zero --max-line-length=120`
4. **AST parse check** — Verify the script can be parsed without executing it

The matrix covers **Python 3.10, 3.11, and 3.12**.

---

## 🗺️ Roadmap

- [x] Gemini AI agents (Opening, Tactical, Positional, Endgame)
- [x] AI Orchestrator with phase detection and multi-agent synthesis
- [x] Loop mode — play `N` games in sequence automatically (`--games N`)
- [x] Elo tracking — estimate the AI's rating over time (`elo_history.json`)
- [x] Fine-tuning pipeline — convert `training_data.pgn` to JSONL (`finetune_pipeline.py`)
- [x] Web dashboard — live board visualisation (`dashboard.py`)
- [x] Support additional LLMs — GPT-4o, Claude, and any future provider (`llm_adapter.py`)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
