#!/usr/bin/env python3
"""Simple LangChain agent using OpenAI and a wizard lookup tool.

Usage:
  - Install: pip install -r requirements.txt
  - Set OPENAI_API_KEY environment variable
  - Run one-off query: python src/agent.py --query "What is 12 * 7?"
  - Start REPL: python src/agent.py --repl
"""

import argparse
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import requests
from langchain.tools import tool
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Constants
WIZARD_API_BASE = "https://wizard-world-api.herokuapp.com"
WIZARD_API_TIMEOUT = 8
MAX_WIZARD_RESULTS = 5


class WizardLookupError(Exception):
    """Raised when wizard lookup fails."""

    pass


def _extract_wizard_info(wizard_data: dict) -> dict:
    """Extract name and potions from wizard API response."""
    name = wizard_data.get("name") or wizard_data.get("firstName") or str(wizard_data)
    potions = wizard_data.get("elixirs") or wizard_data.get("potions") or []
    return {"name": name, "potions": potions}


def _format_potion(potion: dict) -> dict:
    """Format a single potion result."""
    return {
        "potion_name": potion.get("name") or potion.get("title") or "Unknown",
        "potion_description": potion.get("effect") or potion.get("description") or "",
    }


def _wizard_lookup(query: str) -> str:
    """Search the Wizard World API and return potions used by matching wizards.

    Args:
        query: Search term (wizard name, etc.)

    Returns:
        JSON string with query and potions list
    """
    query = query.strip()
    if not query:
        return json.dumps({"query": "", "potions": [], "error": "Please provide a search query"})

    potions_list = []

    try:
        params = {"name": query}
        url = f"{WIZARD_API_BASE}/wizards?{urlencode(params)}"
        response = requests.get(url, timeout=WIZARD_API_TIMEOUT)
        response.raise_for_status()

        wizards = response.json()
        if not wizards:
            return json.dumps({"query": query, "potions": []})

        for wizard in wizards[:MAX_WIZARD_RESULTS]:
            wizard_info = _extract_wizard_info(wizard)
            for potion in wizard_info["potions"]:
                potion_data = _format_potion(potion)
                potions_list.append({"wizard": wizard_info["name"], **potion_data})

    except requests.RequestException as e:
        logger.warning(f"Failed to query wizard API: {e}")
        return json.dumps(
            {"query": query, "potions": [], "error": f"API request failed: {e}"}
        )
    except Exception as e:
        logger.error(f"Unexpected error in wizard lookup: {e}")
        return json.dumps(
            {"query": query, "potions": [], "error": f"Unexpected error: {e}"}
        )

    return json.dumps({"query": query, "potions": potions_list}, ensure_ascii=False)


# LangChain tool wrapper
wizard_lookup = tool(_wizard_lookup)


def _get_llm():
    """Get OpenAI LLM, trying multiple import paths for compatibility."""
    llm_classes = [
        ("langchain.llms", "OpenAI"),
        ("langchain.chat_models", "ChatOpenAI"),
        ("langchain", "OpenAI"),
    ]

    for module_name, class_name in llm_classes:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue

    raise RuntimeError(
        "No compatible OpenAI LLM found. Ensure langchain and langchain-openai are installed."
    )


def _check_api_key() -> None:
    """Verify OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")


class Agent(ABC):
    """Abstract agent interface."""

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute a query and return results."""


class HybridAgent(Agent):
    """Agent that tries direct tool lookup first, then falls back to LLM."""

    def __init__(self, llm_agent: Any):
        self.llm_agent = llm_agent

    def run(self, query: str) -> str:
        """Execute query, preferring direct tool results if available."""
        # Try direct tool lookup first
        tool_result = _wizard_lookup(query)
        try:
            data = json.loads(tool_result)
            if data.get("potions"):
                return tool_result
        except json.JSONDecodeError:
            pass

        # Fall back to LLM agent
        return self._run_llm_agent(query)

    def _run_llm_agent(self, query: str) -> str:
        """Run the underlying LLM agent."""
        if hasattr(self.llm_agent, "run"):
            return self.llm_agent.run(query)

        if hasattr(self.llm_agent, "stream"):
            parts = []
            for chunk in self.llm_agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="updates",
            ):
                parts.append(str(chunk))
            return "".join(parts)

        raise RuntimeError("LLM agent has no compatible interface")


def _build_initialize_agent_style() -> Agent:
    """Build agent using older LangChain initialize_agent API."""
    from langchain.agents import AgentType, initialize_agent

    llm_class = _get_llm()
    llm = llm_class(temperature=0)
    agent = initialize_agent(
        [wizard_lookup],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return HybridAgent(agent)


def _build_create_agent_style() -> Agent:
    """Build agent using newer LangChain create_agent API."""
    from langchain.agents import create_agent

    agent = create_agent(
        model="openai:gpt-3.5-turbo",
        tools=[_wizard_lookup],
        system_prompt="You are a helpful assistant",
    )
    return HybridAgent(agent)


def build_agent() -> Agent:
    """Build and return an agent, trying multiple LangChain versions."""
    _check_api_key()

    # Try newer API first
    try:
        return _build_initialize_agent_style()
    except Exception as e:
        logger.debug(f"initialize_agent style failed: {e}, trying create_agent")

    try:
        return _build_create_agent_style()
    except Exception as e:
        raise RuntimeError(f"Failed to build agent with any available API: {e}")


def format_output(text: str) -> str:
    """Format output, pretty-printing JSON when possible."""
    try:
        data = json.loads(text)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


def run_query(agent: Agent, query: str) -> None:
    """Run a single query and print formatted result."""
    result = agent.run(query)
    print(format_output(result))


def run_repl(agent: Agent) -> None:
    """Run interactive REPL."""
    print("Starting REPL (Ctrl+D or empty line to quit)")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            break

        result = agent.run(user_input)
        print(format_output(result))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple LangChain agent")
    parser.add_argument("--query", "-q", help="Single query to run")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL")
    args = parser.parse_args()

    agent = build_agent()

    if args.query:
        run_query(agent, args.query)
    elif args.repl:
        run_repl(agent)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
