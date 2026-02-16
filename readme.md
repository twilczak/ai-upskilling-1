# Simple LangChain Agent

A LangChain agent that uses OpenAI and the Wizard World API to look up wizards and their potions. Supports both older and newer LangChain APIs.

## Quick Start

1. Create a virtual environment and install dependencies (from workspace root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your OpenAI API key (required to run the agent):

```bash
export OPENAI_API_KEY="sk-..."
```

## Run Examples

Run from the workspace root:

- Single query (wizard lookup):

```bash
python3 src/agent.py --query "Dumbledore"
```

- Interactive REPL:

```bash
python3 src/agent.py --repl
```

## Features

- **Wizard Lookup Tool**: Queries the [Wizard World API](https://wizard-world-api.herokuapp.com/) for wizards and returns potions/elixirs used by matching wizards as structured JSON
- **Multiple LangChain Versions**: Works with both older (`initialize_agent`) and newer (`create_agent`) LangChain APIs
- **Hybrid Agent**: Direct tool lookup when available, falls back to LLM-based agent when needed

## Example Output

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

## Files

- `src/agent.py` — main agent script
- `requirements.txt` — dependency manifest (workspace root)
- `tests/test_agent.py` — test suite
