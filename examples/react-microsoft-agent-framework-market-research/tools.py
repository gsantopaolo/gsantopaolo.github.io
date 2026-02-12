#!/usr/bin/env python3
"""
ReAct Pattern Market Research - Tool Definitions (Microsoft Agent Framework)

This module defines tools for use with Microsoft Agent Framework.
The ReAct pattern interleaves reasoning and action: Think → Act → Observe → repeat.

Tools:
    - web_search: Search the web for current market information
    - calculator: Perform mathematical calculations for metrics

Author: GP (genmind.ch)
License: MIT
Blog Post: https://genmind.ch/posts/Building-ReAct-Agents-with-Microsoft-Agent-Framework-From-Theory-to-Production/
"""

import os
import json
import ast
import operator
from typing import Annotated
import requests


def web_search(
    query: Annotated[str, "The search query to execute"]
) -> str:
    """
    Search the web for current information using Serper API.

    This tool enables the ReAct agent to access real-time market data,
    company information, industry reports, and other web content.

    Args:
        query: Search query string (e.g., "AI agent market size 2024-2026")

    Returns:
        Formatted JSON string with search results including titles, snippets, and URLs

    Raises:
        RuntimeError: If API key is missing or request fails

    Security:
        - 10-second timeout prevents hanging
        - Graceful error handling
        - Structured JSON output

    Examples:
        >>> web_search("Goldman Sachs AI agent deployment")
        >>> web_search("ReAct pattern vs Planning benchmark")
    """
    api_key = os.getenv("SERPER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SERPER_API_KEY environment variable not set. "
            "Get your free API key at https://serper.dev"
        )

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 10})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()

        # Format results for LLM consumption
        formatted_results = []

        # Knowledge graph (if available) - highest quality
        if 'knowledgeGraph' in results:
            kg = results['knowledgeGraph']
            formatted_results.append({
                "type": "knowledge_graph",
                "title": kg.get('title', ''),
                "description": kg.get('description', ''),
                "source": kg.get('source', ''),
                "attributes": kg.get('attributes', {})
            })

        # Organic search results
        for item in results.get('organic', [])[:5]:
            formatted_results.append({
                "title": item.get('title', 'No title'),
                "snippet": item.get('snippet', 'No snippet'),
                "link": item.get('link', ''),
                "position": item.get('position', 0)
            })

        return json.dumps(formatted_results, indent=2)

    except requests.Timeout:
        return f"ERROR: Search timed out after 10 seconds for query: {query}"
    except requests.RequestException as e:
        return f"ERROR: Search failed: {str(e)}"
    except json.JSONDecodeError:
        return "ERROR: Failed to parse search results"


def calculator(
    expression: Annotated[str, "Mathematical expression to evaluate (e.g., '(10 * 5) / 2')"]
) -> str:
    """
    Safely evaluate mathematical expressions using AST parsing.

    Supports basic arithmetic operations: +, -, *, /, **, %
    Uses AST parsing instead of eval() to prevent code injection attacks.

    Args:
        expression: Mathematical expression as string

    Returns:
        Result as string, or error message if evaluation fails

    Security:
        Only allows safe mathematical operations. No arbitrary code execution.
        NEVER use eval() on user input!

    Examples:
        >>> calculator("2 + 2")
        "4.0"
        >>> calculator("((47.1 / 5.43) ** (1/6) - 1) * 100")
        "43.2"
        >>> calculator("192 / 100")
        "1.92"
    """
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(node):
        """Recursively evaluate AST nodes safely."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op = ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(operand)
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval_node(tree.body)
        return str(result)
    except SyntaxError:
        return f"ERROR: Invalid expression: {expression}"
    except ValueError as e:
        return f"ERROR: {str(e)}"
    except ZeroDivisionError:
        return "ERROR: Division by zero"
    except Exception as e:
        return f"ERROR: Calculation failed: {str(e)}"


# Export all tools
__all__ = ['web_search', 'calculator']
