# Contributing to Cyberchess-Dojo

Thank you for taking the time to contribute! 🎉

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
