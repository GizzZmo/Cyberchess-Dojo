# Contributing to Cyberchess-Dojo

Thank you for taking the time to contribute! 🎉

> 📖 For a full overview of the project architecture, configuration options, and agent design, see the **[in-browser Wiki](http://127.0.0.1:5000/wiki)** (run `python dashboard.py` first) or the sections below in this file.

## How to Contribute

### Reporting Bugs
Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) issue template. Include:
- Your OS and Python version
- Stockfish version and path
- Full error output / stack trace

### Requesting Features
Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) issue template.

### Submitting Pull Requests

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install** dev dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flake8
   ```

3. **Make your changes** – keep them focused and small.

4. **Lint** before committing:
   ```bash
   # Hard errors (must be zero)
   flake8 . --select=E9,F63,F7,F82 --show-source
   # Style warnings (review, fix where practical)
   flake8 . --exit-zero --max-line-length=120 --statistics
   ```

5. **Commit** with a clear message:
   ```
   feat: add loop mode to play N games in sequence
   fix: handle Gemini timeout gracefully
   docs: update configuration table in README
   ```

6. **Open a Pull Request** using the provided [PR template](.github/pull_request_template.md).

---

## 🤖 AI Training Pipeline

The repository includes a fully automated training workflow at [`.github/workflows/train.yml`](.github/workflows/train.yml).

### Running the Workflow

Go to **Actions → AI Training Pipeline** → **Run workflow** and choose your parameters:

| Input | Default | Description |
|-------|---------|-------------|
| `games` | `5` | Number of games |
| `llm` | `gemini` | Provider: `gemini`, `openai`, or `claude` |
| `model` | *(provider default)* | Model name override |
| `skill` | `5` | Starting Stockfish skill (0–20) |
| `best_of_n` | `3` | LLM samples per move |
| `time_control` | `rapid` | `classic`, `rapid`, or `lightning` |
| `all_moves` | `false` | Include both colours in the fine-tuning dataset |

The workflow also runs automatically on a weekly schedule (Sundays at 02:00 UTC).

### Required Secrets

Set these in **Settings → Secrets and variables → Actions**:

| Secret | Required for |
|--------|-------------|
| `GOOGLE_API_KEY` | `--llm gemini` |
| `OPENAI_API_KEY` | `--llm openai` |
| `ANTHROPIC_API_KEY` | `--llm claude` |

### What the Workflow Produces

Each run uploads a `training-run-<N>` artefact (90-day retention) containing:
- `training_data.pgn` — all accumulated game records
- `finetune_data.jsonl` — fine-tuning dataset (compatible with OpenAI, Vertex AI, Hugging Face)
- `elo_history.json` — full Elo rating history
- `adaptive_progress.json` — adaptive curriculum snapshots

Training state (Elo, adaptive progress, PGN) is persisted in `actions/cache` so it accumulates across runs.

See [README.md — AI Training Pipeline](README.md#-ai-training-pipeline-trainyml) for the full pipeline diagram and details.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a max line length of **120 characters**.
- Use descriptive variable names.
- Add docstrings to any new functions you introduce.
- Never commit API keys or secrets — always read them from environment variables.

## Commit Message Convention

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring without behaviour change |
| `chore:` | Build / CI / tooling |

## Questions?

Open a [Discussion](https://github.com/GizzZmo/Cyberchess-Dojo/discussions) or an Issue — we're happy to help.

You can also browse the [About page](http://127.0.0.1:5000/about) and [Wiki](http://127.0.0.1:5000/wiki) in the dashboard for project background and full documentation.

---

## 🧩 Extending the Project

### Adding a New Agent

1. Create `agents/your_agent.py` subclassing `BaseChessAgent`:

   ```python
   from agents.base_agent import BaseChessAgent
   import chess

   class YourAgent(BaseChessAgent):
       name = "YourAgent"
       description = "What this agent specialises in"

       def _build_prompt(self, board: chess.Board, legal_moves: list[str]) -> str:
           return f"""You are a chess specialist ...
   Current board (FEN): {board.fen()}
   Legal moves: {', '.join(legal_moves)}
   ...
   On the very last line, write ONLY the UCI move (e.g. e7e5).
   """
   ```

2. Export it from `agents/__init__.py`:
   ```python
   from agents.your_agent import YourAgent
   __all__ = [..., "YourAgent"]
   ```

3. Wire it into `ChessOrchestrator` (`orchestrator.py`) — add it to `__init__` and the phase-routing logic in `get_best_move` / `get_move`.

Key rules for prompts:
- Always include the FEN and the full legal move list.
- Instruct the model to put **only** the UCI move on the very last line.
- Keep the domain focus tight — a specialist beats a generalist in its own area.

---

### Adding a New LLM Provider

1. Add a new adapter class in `llm_adapter.py` that extends `BaseLLMAdapter`:

   ```python
   class MyProviderAdapter(BaseLLMAdapter):
       def __init__(self, model_name: str = "default-model", api_key: str = None):
           # initialise the SDK client here
           ...

       def generate_content(self, prompt: str) -> _LLMResponse:
           # call the provider's API and return _LLMResponse(text)
           ...

       @property
       def model_name(self) -> str:
           return self._model_name
   ```

2. Register the provider in `_PROVIDER_DEFAULTS` and the `create_adapter` factory:

   ```python
   _PROVIDER_DEFAULTS = {
       ...,
       "myprovider": "default-model",
   }

   # in create_adapter():
   if provider == "myprovider":
       return MyProviderAdapter(model_name=resolved_model, api_key=api_key)
   ```

3. Add `"myprovider"` to the `--llm` choices in `cyberchess.py`'s `_build_arg_parser`.

The adapter only needs to expose `generate_content(prompt) -> _LLMResponse` (duck-typed); everything else — agents, orchestrator, best-of-N — works without modification.
