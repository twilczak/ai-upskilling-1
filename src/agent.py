#!/usr/bin/env python3
"""Simple LangChain agent using OpenAI and a safe calculator tool.

Usage:
  - Install: pip install -r requirements.txt
  - Set OPENAI_API_KEY environment variable
  - Run one-off query: python src/agent.py --query "What is 12 * 7?"
  - Start REPL: python src/agent.py --repl
"""

import os
import argparse

from langchain.tools import tool

OpenAI = None


def _wizard_lookup(query: str) -> str:
    """Search the Wizard World API for the given query and return short results.

    The function queries a few common endpoints (`wizards`, `spells`, `houses`) by
    name and returns the first matching items found. Returns an informative
    message if nothing is found or on errors.
    """
    import requests
    from urllib.parse import urlencode

    import json

    base = "https://wizard-world-api.herokuapp.com"
    # Limit to wizards only since we want potions/elixirs used by a wizard
    endpoints = ["wizards"]
    q = query.strip()
    if not q:
        return "Please provide a search query (e.g. a wizard or spell name)."

    potions_list = []
    for ep in endpoints:
        try:
            params = {"name": q}
            url = f"{base}/{ep}?{urlencode(params)}"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data:
                continue
            # Limit to first 5 wizard results
            for item in data[:5]:
                name = item.get("name") or item.get("firstName") or str(item)
                # Some wizard records include `elixirs` or `potions` fields.
                potions = item.get("elixirs") or item.get("potions") or []
                for p in potions:
                    p_name = p.get("name") or p.get("title") or str(p)
                    p_desc = p.get("effect") or p.get("description") or ""
                    potions_list.append({
                        "wizard": name,
                        "potion_name": p_name,
                        "potion_description": p_desc,
                    })
        except Exception as e:
            # Include errors as entries so caller can inspect
            potions_list.append({"error": f"Error querying {ep}: {e}"})
    if not potions_list:
        return json.dumps({"query": q, "potions": []})

    return json.dumps({"query": q, "potions": potions_list}, ensure_ascii=False)


# LangChain tool wrapper (kept as `wizard_lookup` for agent use)
wizard_lookup = tool(_wizard_lookup)


def build_agent():
    # Import LLMs and agent initializer lazily to allow module import without an
    # available OpenAI model class (helps unit/smoke tests for tools).
    global OpenAI
    try:
        from langchain.llms import OpenAI as _OpenAI  # older style
        OpenAI = _OpenAI
    except Exception:
        try:
            from langchain.chat_models import ChatOpenAI as _ChatOpenAI
            OpenAI = _ChatOpenAI
        except Exception:
            try:
                from langchain import OpenAI as _OpenAI_pkg
                OpenAI = _OpenAI_pkg
            except Exception:
                OpenAI = None

    # Prefer the older `initialize_agent` API if available; otherwise use
    # `create_agent` which is used in more recent LangChain releases.
    def _make_executor(fallback_agent):
        import json

        class Executor:
            def __init__(self, fallback):
                self.fallback = fallback

            def run(self, query: str) -> str:
                # First, call the wizard lookup tool directly and prefer its JSON output
                try:
                    tool_out = _wizard_lookup(query)
                    # tool_out is a JSON string; try to parse and check 'potions'
                    parsed = json.loads(tool_out)
                    if parsed.get("potions"):
                        # Return the raw JSON string so callers get structured data
                        return tool_out
                except Exception:
                    # If the direct tool call fails or returns no potions, fall back
                    pass

                # Fall back to the LLM-based agent/graph
                if hasattr(self.fallback, "run"):
                    return self.fallback.run(query)
                # Some agents expect a messages dict
                if hasattr(self.fallback, "stream"):
                    parts = []
                    for chunk in self.fallback.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="updates"):
                        parts.append(str(chunk))
                    return "".join(parts)
                raise RuntimeError("Fallback agent has no runnable interface")

        return Executor(fallback_agent)

    try:
        from langchain.agents import initialize_agent, AgentType

        if OpenAI is None:
            raise RuntimeError(
                "No compatible OpenAI LLM class found in LangChain; ensure a supported langchain version is installed"
            )

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY environment variable before running")

        llm = OpenAI(temperature=0)
        tools = [wizard_lookup]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
        return _make_executor(agent)
    except Exception:
        from langchain.agents import create_agent

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY environment variable before running")

        # create_agent accepts a model string like "openai:gpt-3.5-turbo"
        graph = create_agent(model="openai:gpt-3.5-turbo", tools=[_wizard_lookup], system_prompt="You are a helpful assistant")

        return _make_executor(graph)


def main():
    parser = argparse.ArgumentParser(description="Simple LangChain agent")
    parser.add_argument("--query", "-q", help="Single query to run")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL")
    args = parser.parse_args()

    agent = build_agent()

    import json

    def _print_maybe_json(text: str):
        try:
            parsed = json.loads(text)
            # Pretty-print structured JSON
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            print(text)

    if args.query:
        _print_maybe_json(agent.run(args.query))
    elif args.repl:
        print("Starting REPL. Empty line to quit.")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                break
            _print_maybe_json(agent.run(text))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
