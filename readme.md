# Simple LangChain Agent

This small example shows how to create a minimal LangChain agent in Python that uses OpenAI as the LLM and exposes a simple safe `calculator` tool.

Setup

1. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Run

- Single query:

```bash
python src/agent.py --query "What is 12 * 7?"
```

- Interactive REPL:

```bash
python src/agent.py --repl
```

Files

- [src/agent.py](simple_langchain_agent/src/agent.py) — main agent script
- [requirements.txt](simple_langchain_agent/requirements.txt) — packages to install

Notes

- This example requires an OpenAI API key set in `OPENAI_API_KEY`.
- The `calculator` tool evaluates arithmetic expressions using Python's AST for safety; it's limited to numeric operations.
