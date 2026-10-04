# Agentic AI Samples

Runnable code behind Soubh's weekly LinkedIn posts on agentic AI. Every post comes with a small project you can run on your own machine in a few minutes, with tests.

## Samples

| Date | Topic | Code | LinkedIn post |
|---|---|---|---|
| 2026-10-02 | Jev vs LLM | [2026-10-02-jev-vs-llm](2026/2026-10-02-jev-vs-llm) | Coming soon |
| 2026-10-02 | How an AI agent uses MCP | [2026-10-02-how-agents-use-mcp](2026/2026-10-02-how-agents-use-mcp) | Coming soon |

## Run a sample

Each folder is a self-contained Python project managed by [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Soubh04/agentic-ai-samples.git
cd agentic-ai-samples/2026/<folder>
uv run pytest -q
```

The folder's own README shows the command that runs the demo. Samples run offline with mock data by default; live modes read API keys from environment variables only.

## How this repository is organised

- One folder per post under its year, named `<date>-<topic>`. Folder names never change once a post is live, so links keep working.
- Each folder has its own dependencies, lock file, README and tests. Nothing is shared between folders, so you can copy one on its own.
- Pull requests run the tests of the folders they change, and every sample is re-tested monthly to catch SDK and API changes.

## Notes

- These are teaching samples, not production code.
- Speed and price figures quoted from vendors are labelled as vendor figures in each sample. Measure on your own workload.
- No real customer data or credentials appear anywhere in this repository.

## License

[MIT](LICENSE)
