# Simple LangChain Agent

A minimal LangChain agent example that wraps a safe arithmetic `calculator` tool and supports both older and newer LangChain APIs.

Quick start

1. Create a virtual environment and install dependencies from the workspace root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your OpenAI API key (required to run the agent):

```bash
export OPENAI_API_KEY="sk-..."
```

Run examples (from workspace root)

- Single query:

```bash
.venv/bin/python src/agent.py --query "What is 12 * 7?"
```

- Interactive REPL:

```bash
.venv/bin/python src/agent.py --repl
```

Notes

- The agent entrypoint is `src/agent.py` (moved up one level from earlier examples).
- Use the top-level `requirements.txt` to install dependencies; it includes `langchain`, `openai`, and `langchain-openai`.
- The code attempts to work with multiple LangChain versions:
	- If `initialize_agent` is available the script will use it.
	- Otherwise it falls back to `create_agent` and runs the resulting graph via a small wrapper.
- The `calculator` tool evaluates arithmetic expressions using Python's AST (safe subset) and supports basic binary and unary numeric operators.
- On macOS with Python 3.14 you may see deprecation warnings from some LangChain internals; if you run into compatibility problems, try Python 3.11 or 3.10.

- The agent now exposes a `wizard_lookup` tool which queries the Wizard World API (https://wizard-world-api.herokuapp.com/) for `wizards`, `spells`, and `houses` by name. The tool will return short, formatted results for matches.
 - The agent now exposes a `wizard_lookup` tool which queries the Wizard World API (https://wizard-world-api.herokuapp.com/) for `wizards` and returns potions/elixirs used by matching wizards as structured JSON.

Example JSON output:

```json
{
	"query": "Dumbledore",
	"potions": [
		{
			"wizard": "Albus Dumbledore",
			"potion_name": "First Love Beguiling Bubbles",
			"potion_description": "[effect or description]"
		}
	]
}
```

Files

- `src/agent.py` — main agent script
- `requirements.txt` — dependency manifest (workspace root)

If you'd like, I can also add a small test script or update the top-level project README to link this example.
