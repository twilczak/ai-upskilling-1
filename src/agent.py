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

try:
    from langchain.llms import OpenAI
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import tool
except Exception:
    raise RuntimeError("Please install requirements from simple_langchain_agent/requirements.txt")


@tool
def calculator(expression: str) -> str:
    import ast, operator as op
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.USub: op.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
        raise ValueError("Unsupported expression")

    try:
        node = ast.parse(expression, mode="eval").body
        return str(_eval(node))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def build_agent():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable before running")
    llm = OpenAI(temperature=0)
    tools = [calculator]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Simple LangChain agent")
    parser.add_argument("--query", "-q", help="Single query to run")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL")
    args = parser.parse_args()

    agent = build_agent()

    if args.query:
        print(agent.run(args.query))
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
            print(agent.run(text))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
