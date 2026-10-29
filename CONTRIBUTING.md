# Contributing

Thank you for helping improve these samples. The conventions below keep the history readable and every sample runnable on its own.

## Branches and pull requests

- Every change reaches `main` through a pull request. Do not push to `main` directly.
- Name the branch after the change: `add/<date>-<topic>` for a new sample, `fix/<topic>` for a defect, `docs/<topic>` for documentation only.
- Keep a pull request to one sample or one change, and under about 400 changed lines where the work allows. Small pull requests are reviewed faster and are easier to revert.
- The `tests` workflow runs the tests of every sample folder a pull request changes. A pull request is merged only when it passes.

## Commit messages

Start every commit subject with a type, followed by a colon and a short description in the imperative:

| Type | Use it for |
|---|---|
| `feat` | A new sample or new behaviour in a sample |
| `fix` | A defect in a sample or its tests |
| `docs` | README or documentation changes only |
| `test` | Tests only |
| `refactor` | A code change that does not change behaviour |
| `chore` | Dependencies, lock files, tooling |
| `revert` | Undoing an earlier commit (Git's own `Revert "…"` subject is also fine) |

Example: `fix: keep deleted and expired facts out of an open session`.

## Samples

- One folder per post under its year, named `<date>-<topic>`. A folder name never changes once its post is live, so links keep working.
- Each folder is a self-contained Python project managed by [uv](https://docs.astral.sh/uv/), with its own dependencies, lock file, README and tests. Nothing is shared between folders.
- Tests run offline, without credentials. Samples that call cloud services read keys from environment variables only; never commit a key, token or `.env` file.
- Use synthetic data only. Never add real personal, customer or financial data.

## Repository hygiene

- `.gitignore` keeps `.claude/`, `.mcp.json` and `Archive/` out of the repository.
- Run the sample's tests locally (`uv run pytest`) before you open a pull request.
