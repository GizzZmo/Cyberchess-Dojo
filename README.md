# ♟️ Cyberchess-Dojo

[![CI](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/ci.yml/badge.svg)](https://github.com/GizzZmo/Cyberchess-Dojo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/GizzZmo/Cyberchess-Dojo)](https://github.com/GizzZmo/Cyberchess-Dojo/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **An AI Training Arena** — Stockfish (The Teacher) plays chess against a Gemini-powered multi-agent system (The Student), generating PGN training data for future fine-tuning.

---

## 🧠 Concept

Cyberchess-Dojo is an **automated chess training pipeline** where a classical engine and a large language model compete against each other:

| Role | Engine | Colour |
|------|--------|--------|
| 🎓 Teacher | [Stockfish](https://stockfishchess.org/) | White |
| 🤖 Student | [Gemini 1.5 Flash](https://ai.google.dev/) via AI Orchestrator | Black |

Every game is saved as a [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation) file (`training_data.pgn`). The long-term goal is to use this dataset to **fine-tune Gemini** so it learns from Stockfish's play.

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
│            synthesise (if agents disagree)           │
│                   │                                  │
│            UCI move ──► board.push()                 │
│                            │                         │
│                     training_data.pgn                │
└──────────────────────────────────────────────────────┘
```

---

## 🤖 AI Agents & Orchestrator

### Specialised Agents (`agents/`)

Each agent is a focused Gemini persona with a domain-specific prompt:

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

When two agents are consulted and they **disagree**, the orchestrator makes a third Gemini call — acting as a grandmaster arbitrator — to synthesise a final decision from both analyses.

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | |
| [Stockfish](https://stockfishchess.org/download/) | ≥ 15 | Must be installed separately |
| Google Gemini API key | — | [Get one free](https://aistudio.google.com/app/apikey) |

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
export STOCKFISH_PATH="/usr/local/bin/stockfish"   # adjust to your path
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

**Windows (PowerShell)**
```powershell
$env:STOCKFISH_PATH = "C:\Users\Jon\Downloads\stockfish\stockfish-windows-x86-64.exe"
$env:GOOGLE_API_KEY = "your-gemini-api-key-here"
```

### 4. Run the arena

```bash
python cyberchess.py
```

The game will be printed move-by-move to the terminal and appended to `training_data.pgn` when it finishes.

---

## ⚙️ Configuration

All knobs are at the top of `cyberchess.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `STOCKFISH_PATH` | env / `"YOUR_STOCKFISH_PATH_HERE"` | Path to the Stockfish binary |
| `GOOGLE_API_KEY` | env / `"YOUR_GEMINI_API_KEY_HERE"` | Gemini API key |
| `STOCKFISH_SKILL_LEVEL` | `5` | Stockfish strength 0 (weakest) – 20 (GM) |
| `STOCKFISH_TIME_LIMIT` | `0.1` | Seconds Stockfish spends per move |
| `GEMINI_MODEL_NAME` | `"gemini-1.5-flash"` | Gemini model to use |

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
├── orchestrator.py             # ChessOrchestrator — routes board states to agents
├── cyberchess.py               # Main arena script
├── requirements.txt            # Python dependencies
├── training_data.pgn           # Generated — game records for fine-tuning
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
- [ ] Loop mode — play `N` games in sequence automatically
- [ ] Elo tracking — estimate Gemini's rating over time
- [ ] Fine-tuning pipeline — use `training_data.pgn` to adapt the model
- [ ] Web dashboard — live board visualisation
- [ ] Support additional LLMs (GPT-4o, Claude, etc.)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

