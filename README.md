# ♟️ Cyberchess-Dojo

[![CI](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/ci.yml/badge.svg)](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/ci.yml)
[![AI Training Pipeline](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/train.yml/badge.svg)](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/train.yml)
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
│   training_data.pgn  elo_history.json  settings.json │
└──────────────────────────────────────────────────────┘
```

---

## 📸 Screenshots

### Dashboard — Live Board
![Cyberchess Dojo dashboard showing the live board, Elo rating panel, move list, and current game info cards](https://github.com/user-attachments/assets/f013b802-4f1d-4090-b847-039f68b4203f)

### Welcome Menu — Time Controls
![Welcome menu modal with Classic, Rapid, and Lightning time control options](https://github.com/user-attachments/assets/ca249d3a-c766-46c5-8b3b-d2fcd7336aa1)

### Settings Modal
![Settings modal showing Stockfish skill slider, LLM provider selector, and API key fields](https://github.com/user-attachments/assets/730d2b92-847f-4e58-90a2-e14f7dc76a26)

### About Page
![About page with project description, key features grid, and technology stack table](https://github.com/user-attachments/assets/15bf432a-0217-4298-9855-90ed05698124)

### Wiki Page
![Wiki page with sidebar navigation, prerequisites table, and installation instructions](https://github.com/user-attachments/assets/165b4bc1-701e-427a-b1c3-413c10f3df5d)

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

Additional selection safeguards now applied by the orchestrator:

- **Opening fast-path**: in opening phase, the orchestrator first checks Polyglot/embedded theory and immediately plays the top legal theory move when available.
- **Tactical safety filter**: after ranking candidates, obviously riskier options can be replaced by a materially safer alternative from the same candidate set.
- **Endgame conversion filter**: in endgames, candidate moves are re-scored for king activity and passed-pawn conversion potential before final selection.

When candidate analyses disagree, the orchestrator still makes a final grandmaster-style ranking call to synthesise the strongest move.

---

## 🔄 How It Works

A full game loop looks like this:

1. **Startup** — `cyberchess.py` validates config, creates the LLM adapter, and loads Elo history.
2. **Per-game** — Stockfish (White) and the LLM orchestrator (Black) alternate moves until the game is over.
3. **Per-move (Black)** —
   - The `ChessOrchestrator` detects the game phase (opening / middlegame / endgame).
   - In the opening, it first attempts a direct theory move from Polyglot/embedded opening knowledge.
   - Otherwise, it selects the appropriate specialist agent(s) and requests `N` independent move samples (best-of-N).
   - If all samples agree, that move is played immediately.
   - If samples differ, a ranking call picks the strongest candidate.
   - Tactical/endgame post-filters can replace clearly inferior choices with safer or more convertible alternatives.
   - Before each move the arena checks `pause_flag.json`; if a pause was requested from the dashboard it waits until the flag clears.
4. **Post-game** — The completed game is appended to `training_data.pgn`; the AI's Elo is updated in `elo_history.json`.
5. **Fine-tuning** — After collecting games, `finetune_pipeline.py` converts the PGN into a JSONL dataset ready for supervised fine-tuning.

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | |
| [Stockfish](https://stockfishchess.org/download/) | ≥ 15 | Must be installed separately |
| LLM API key | — | See [LLM Provider Setup](#-llm-provider-setup) |

---

## 🔑 Environment Variables

All sensitive configuration is read from environment variables (never hard-coded).  API keys can also be entered directly in the **Settings** panel of the web dashboard (see [Web Dashboard](#-web-dashboard)) and are stored locally in `settings.json`.

| Variable | Provider | Required | Description |
|----------|----------|----------|-------------|
| `STOCKFISH_PATH` | — | ✅ | Full path to the Stockfish binary |
| `GOOGLE_API_KEY` | Gemini | ✅ (Gemini only) | Google AI Studio API key |
| `OPENAI_API_KEY` | OpenAI | ✅ (OpenAI only) | OpenAI platform API key |
| `ANTHROPIC_API_KEY` | Claude | ✅ (Claude only) | Anthropic API key |
| `PGN_FILE` | Dashboard | — | Override the default `training_data.pgn` path |
| `ELO_FILE` | Dashboard | — | Override the default `elo_history.json` path |
| `STATE_FILE` | Dashboard | — | Override the default `game_state.json` path |
| `PAUSE_FILE` | Dashboard | — | Override the default `pause_flag.json` path |
| `SETTINGS_FILE` | Dashboard | — | Override the default `settings.json` path |

CLI flags (`--api-key`, `--stockfish`) override the corresponding environment variables.

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
$env:STOCKFISH_PATH = "C:/Users/Jon/Downloads/stockfish/stockfish-windows-x86-64.exe"
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

> **Tip:** API keys can also be set from the browser using the **⚙ Settings** button in the web dashboard — no terminal needed.

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
                     [--time-control {classic,rapid,lightning}]
                     [--matchup {stockfish-ai,stockfish-stockfish,ai-ai,ai-stockfish}]
                     [--llm {gemini,openai,claude}] [--model MODEL_NAME]
                     [--api-key KEY] [--best-of-n N]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--games N` | `1` | Number of games to play in sequence (loop mode) |
| `--skill 0-20` | `5` | Stockfish strength 0 (weakest) – 20 (Grandmaster) |
| `--time SECS` | `0.1` | Seconds Stockfish spends per move |
| `--time-control` | `rapid` | Time-control preset: `classic`, `rapid`, or `lightning` |
| `--matchup` | `stockfish-ai` | Player combination: `stockfish-ai`, `stockfish-stockfish`, `ai-ai`, or `ai-stockfish` |
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

## 🧭 Adaptive Training

When exactly one side is AI (`stockfish-ai` or `ai-stockfish`), `adaptive_system.py` adjusts the next game's challenge plan from recent results:

- **Recovery regime** (recent score low): slightly reduces Stockfish strength/time and increases best-of-N.
- **Challenge regime** (recent score high): slightly increases Stockfish strength/time and increases best-of-N.
- **Stable regime**: keeps the base settings.

To reduce oscillation near thresholds, the adaptive manager now uses:

- **Exponential smoothing** of recent performance.
- **Hysteresis thresholds** so regime switches require clearer evidence.

The generated plan is printed each game as:

```
🧭 Adaptive plan: regime=... | recent_score=... | skill=... | time=...s | best_of_n=...
```

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

> **Alternative:** API keys for all three providers can be entered without a terminal via the **⚙ Settings** panel in the web dashboard.

---

## 🌐 Web Dashboard

The dashboard provides a live browser view of the board, move list, Elo history chart, and game log, with an interactive cyberpunk / matrix aesthetic.

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

### Dashboard pages

| Page | URL | Description |
|------|-----|-------------|
| **Dashboard** | `/` | Live board, move list, Elo chart, game log, pause/resume, PGN import/export, and Settings modal |
| **About** | `/about` | Project overview, features, architecture, and links |
| **Wiki** | `/wiki` | Full documentation — setup, CLI reference, agents, FAQ |

### Dashboard features

#### 🧭 Welcome Menu (Time Controls)
On first dashboard load, a welcome menu lets you select a chess pace:

- **Classic** — 0.30s per Stockfish move
- **Rapid** — 0.10s per Stockfish move
- **Lightning** — 0.03s per Stockfish move

The choice is stored in `settings.json` as `time_control_mode` and is used by the next arena run unless you explicitly pass `--time`.

#### ⏸ Pause / Resume
Click the **Pause** button on the Live Board card to freeze the arena between moves.  The arena polls `pause_flag.json` before every move and waits until the flag clears.  Click **Resume** to continue.

#### 📋 Move List
A two-column SAN move table (White / Black) updates after every half-move and automatically scrolls to show the latest move.

#### ⬆ Import PGN
Upload a `.pgn` file to append its games to `training_data.pgn`.  The file is validated (must contain at least one readable game; maximum 10 MB) before being written.

#### ⬇ Export PGN
Downloads the full `training_data.pgn` as `cyberchess_games.pgn`.

#### ⚙ Settings
Click the **⚙ Settings** button in the navigation bar to open the settings modal.  Changes are persisted to `settings.json` and take effect on the next arena run.

| Setting | Description |
|---------|-------------|
| Stockfish skill | Skill level 0–20 (slider) |
| Time per move | Seconds Stockfish is allowed per move |
| Time control mode | Classic, Rapid, or Lightning preset |
| LLM provider | Gemini, OpenAI, or Anthropic Claude |
| Model name | Model identifier override |
| Best-of-N | Number of LLM samples per move |
| Gemini API key | Google AI Studio key |
| OpenAI API key | OpenAI platform key |
| Anthropic API key | Anthropic key |

> API key fields are stored locally in `settings.json` and are partially masked when loaded back into the browser (`****xxxx`).

### Dashboard API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | — | Live dashboard HTML |
| `GET /about` | — | About page |
| `GET /wiki` | — | Wiki documentation |
| `GET /api/state` | — | Current board state (FEN, phase, last move, SAN move list) |
| `GET /api/elo` | — | Full Elo history JSON |
| `GET /api/games` | — | Completed games parsed from `training_data.pgn` |
| `GET /api/games/export` | — | Download `training_data.pgn` |
| `POST /api/games/import` | multipart/form-data (`file`) | Append uploaded PGN to training dataset |
| `GET /api/pause` | — | Return current pause state `{"paused": bool}` |
| `POST /api/pause` | JSON `{"paused": bool}` | Set or toggle pause flag |
| `GET /api/settings` | — | Return current settings (API keys partially masked) |
| `POST /api/settings` | JSON | Persist settings to `settings.json` |

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
│       ├── ci.yml              # Lint + syntax check on every push/PR
│       └── train.yml           # Automated AI training pipeline (manual + weekly schedule)
├── agents/
│   ├── __init__.py             # Package exports
│   ├── base_agent.py           # Shared base class (retry, UCI extraction, fallback)
│   ├── opening_agent.py        # Opening principles specialist
│   ├── tactical_agent.py       # Tactics specialist (checks, captures, forks)
│   ├── positional_agent.py     # Positional / strategic specialist
│   └── endgame_agent.py        # Endgame technique specialist
├── templates/
│   ├── index.html              # Web dashboard — live board, move list, settings, import/export
│   ├── about.html              # About page — project overview, features, architecture
│   └── wiki.html               # Wiki — full in-browser documentation
├── orchestrator.py             # ChessOrchestrator — routes board states to agents
├── cyberchess.py               # Main arena script (loop mode, pause support, Elo, dashboard)
├── llm_adapter.py              # Unified LLM interface (Gemini, OpenAI, Claude)
├── elo_tracker.py              # Elo rating system with JSON persistence
├── finetune_pipeline.py        # PGN → JSONL fine-tuning dataset generator
├── dashboard.py                # Flask web dashboard server (settings, pause, import/export)
├── requirements.txt            # Python dependencies
├── training_data.pgn           # Generated — game records for fine-tuning
├── elo_history.json            # Generated — Elo rating history
├── game_state.json             # Generated — live board state for dashboard
├── pause_flag.json             # Generated — pause/resume flag written by dashboard
├── settings.json               # Generated — persisted UI settings (API keys, skill level, …)
├── finetune_data.jsonl         # Generated — fine-tuning dataset
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 🛠️ CI / Workflow

### Lint CI (`ci.yml`)

The [CI workflow](.github/workflows/ci.yml) runs on every push and pull request to `main`/`master`:

1. **Install** — `pip install -r requirements.txt` + `flake8`
2. **Syntax errors** — `flake8 --select=E9,F63,F7,F82` (hard fail)
3. **Style warnings** — `flake8 --exit-zero --max-line-length=120`
4. **AST parse check** — Verify the script can be parsed without executing it

The matrix covers **Python 3.10, 3.11, and 3.12**.

---

## 🤖 AI Training Pipeline (`train.yml`)

The [training workflow](.github/workflows/train.yml) automates end-to-end AI training: it plays games, tracks Elo, adapts difficulty, and generates a fine-tuning dataset — all with persistent state across runs.

### Triggers

| Trigger | Description |
|---------|-------------|
| **Manual** (`workflow_dispatch`) | Run from the **Actions** tab with full control over every parameter |
| **Weekly schedule** | Automatically every Sunday at 02:00 UTC with default settings |

### Configurable Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `games` | `5` | Number of games to play |
| `llm` | `gemini` | LLM provider: `gemini`, `openai`, or `claude` |
| `model` | *(provider default)* | Model name override (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `skill` | `5` | Starting Stockfish skill level (0–20) |
| `best_of_n` | `3` | LLM samples per move for best-of-N selection |
| `time_control` | `rapid` | Time-control preset: `classic`, `rapid`, or `lightning` |
| `all_moves` | `false` | Include White's moves in fine-tuning data (default: Black only) |

### Required Secrets

Set these in **Settings → Secrets and variables → Actions** for your repository:

| Secret | Required for |
|--------|-------------|
| `GOOGLE_API_KEY` | `--llm gemini` |
| `OPENAI_API_KEY` | `--llm openai` |
| `ANTHROPIC_API_KEY` | `--llm claude` |

Only the secret for the chosen provider needs to be set. The workflow validates this before running and fails fast with a clear error message if the secret is missing.

### What It Does

```
1. Restore training state from cache
   (elo_history.json, adaptive_progress.json, training_data.pgn)
              ↓
2. Install Stockfish (apt)
              ↓
3. Validate API key secret
              ↓
4. python cyberchess.py --games N --llm ... --skill ... (+ adaptive curriculum)
              ↓
5. python finetune_pipeline.py → finetune_data.jsonl
              ↓
6. Write GitHub Step Summary
   (Elo table, adaptive plan, dataset stats)
              ↓
7. Save updated state back to cache
              ↓
8. Upload artefacts (PGN, JSONL, Elo history, adaptive progress)
```

### Training State Persistence

The workflow uses `actions/cache` with branch-scoped keys so **Elo ratings and the adaptive curriculum carry over across runs**. Each new run restores the most recent cached state (PGN, Elo history, adaptive progress) before playing new games, then saves the updated state when done.

### Concurrency Guard

Only **one training run per branch** can execute at a time (`cancel-in-progress: false` ensures an in-flight run completes before a new one starts).

### Artefacts

Every run uploads a `training-run-<N>` artefact (90-day retention) containing:

- `training_data.pgn` — accumulated game records
- `finetune_data.jsonl` — fine-tuning dataset
- `elo_history.json` — full Elo rating history
- `adaptive_progress.json` — adaptive curriculum snapshots

### Step Summary

After each run, the **GitHub Step Summary** shows:

- Run parameters table
- Current Elo rating and last-10-games history table
- Last adaptive curriculum plan (regime, skill, time, best-of-N)
- Fine-tuning dataset statistics (total examples, by colour)

### Quick Start — Running the Workflow Manually

1. Go to **Actions → AI Training Pipeline** in the GitHub repository.
2. Click **Run workflow**.
3. Fill in the inputs (or leave defaults).
4. Click **Run workflow** to start.

> **Tip:** For the very first run, set `games` to a small number (e.g. `3`) to confirm your secrets are configured correctly.

---

## 📖 Opening Book

The `opening_book.py` module enriches the `OpeningAgent` with structured chess knowledge:

- **ECO database** — ~150 key positions mapped to ECO codes (A00–E99) and opening names.
- **Embedded theory** — Main-line book moves for Black in ~50 key positions (no external file required).
- **Polyglot binary book** — If a `.bin` file is available, the agent reads moves from it first (set `POLYGLOT_PATH` env var or pass the path to `get_book_moves`). Falls back to the embedded theory if the file is absent.
- **Historical game references** — ~100 curated famous games organised by opening family, included in the agent's prompt as illustrative examples.

The `TacticalAgent`, `PositionalAgent`, and `EndgameAgent` similarly draw on `FAMOUS_GAMES` references for tactical masterpieces, positional classics, and endgame studies respectively.

---

## 📂 Generated Files

These files are created automatically during a run and are **not** committed to the repository:

| File | Created by | Description |
|------|-----------|-------------|
| `training_data.pgn` | `cyberchess.py` | Accumulates completed games in PGN format for fine-tuning |
| `elo_history.json` | `elo_tracker.py` | Persists the AI's Elo rating and per-game history across runs |
| `game_state.json` | `cyberchess.py --dashboard` | Live board state (FEN, move list, pause flag) polled by the web dashboard |
| `pause_flag.json` | Dashboard `POST /api/pause` | Pause/resume signal written by the dashboard, read by `cyberchess.py` |
| `settings.json` | Dashboard `POST /api/settings` | Persisted UI settings: skill level, LLM provider, API keys, etc. |
| `adaptive_progress.json` | `adaptive_system.py` | Adaptive curriculum snapshots used to tune challenge over time |
| `finetune_data.jsonl` | `finetune_pipeline.py` | Fine-tuning dataset generated from `training_data.pgn` |

---

## 🗺️ Roadmap

- [x] Gemini AI agents (Opening, Tactical, Positional, Endgame)
- [x] AI Orchestrator with phase detection and multi-agent synthesis
- [x] Loop mode — play `N` games in sequence automatically (`--games N`)
- [x] Elo tracking — estimate the AI's rating over time (`elo_history.json`)
- [x] Fine-tuning pipeline — convert `training_data.pgn` to JSONL (`finetune_pipeline.py`)
- [x] Web dashboard — live board visualisation (`dashboard.py`)
- [x] Support additional LLMs — GPT-4o, Claude, and any future provider (`llm_adapter.py`)
- [x] In-browser About & Wiki pages — project overview and full documentation (`/about`, `/wiki`)
- [x] Pause / Resume — freeze the arena mid-game from the browser
- [x] Move list — live SAN move table on the dashboard
- [x] PGN import / export — upload or download game files from the browser
- [x] Settings panel — configure all options and enter API keys via the browser UI
- [x] Matrix cyberpunk aesthetic — animated matrix rain, neon glow palette, CRT scanline overlay
- [x] Automated AI training pipeline — `train.yml` runs games, tracks Elo, adapts difficulty, and generates a fine-tuning dataset on a weekly schedule or on demand

---

## 🔧 Troubleshooting

**Stockfish not found**
```
ValueError: STOCKFISH_PATH is not set.
```
Set the environment variable pointing to your Stockfish binary, or pass `--stockfish PATH`:
```bash
# Linux / macOS — installed via package manager
export STOCKFISH_PATH="$(which stockfish)"

# Windows — downloaded manually
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
```

**Missing API key**
```
ValueError: GOOGLE_API_KEY is not set.
```
Export the correct key for your chosen provider (see [Environment Variables](#-environment-variables)), or enter it in the **⚙ Settings** modal in the web dashboard.

**`ImportError` for openai / anthropic**
The `openai` and `anthropic` packages are optional.  Install them only if you need those providers:
```bash
pip install openai       # for --llm openai
pip install anthropic    # for --llm claude
```

**LLM keeps playing illegal moves**
- Try a stronger model (`--model gpt-4o`, `--model gemini-1.5-pro`).
- Increase best-of-N sampling (`--best-of-n 5`) so the ranking call has more candidates to choose from.
- The agents retry up to 3 times per move and fall back to a random legal move if all retries fail.

**Dashboard shows a stale board**
- Make sure `cyberchess.py` was started with `--dashboard` so it writes `game_state.json`.
- The dashboard polls every 2 seconds; a slight delay is normal.
- Check that both processes are running in the same working directory so they use the same `game_state.json`.

**Game appears stuck / pause button has no effect**
- The arena only checks the pause flag **between** moves, so it may take a few seconds to respond.
- Verify both the arena and dashboard are running in the same working directory (same `pause_flag.json`).

**Elo resets to 1200 after a run**
`elo_history.json` is written to the current working directory.  Run both `cyberchess.py` and `finetune_pipeline.py` from the same directory, or set `ELO_FILE` to an absolute path.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
