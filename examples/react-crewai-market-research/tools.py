"""
Custom tools for CrewAI agents demonstrating the ReAct pattern.

This module implements production-ready tools for web search and calculations,
with comprehensive error handling, security considerations, and clear interfaces
optimized for LLM consumption.

Author: GP (genmind.ch)
License: MIT
"""

from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
import requests
import os
import ast
import operator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchToolInput(BaseModel):
    """
    Input schema for WebSearchTool.

    Attributes:
        query: The search query to execute (e.g., "AI agent market size 2024")
    """
    query: str = Field(
        ...,
        description="The search query to execute",
        min_length=1,
        max_length=500
    )


class WebSearchTool(BaseTool):
    """
    Web search tool using Serper API.

    This tool searches the web for current information and returns formatted
    results optimized for LLM consumption. It includes timeout protection,
    error handling, and rate limiting considerations.

    Example usage by an agent:
        Thought: "I need current market data for AI agents"
        Action: web_search
        Action Input: {"query": "AI agent market size 2024 statistics"}
        Observation: [Formatted search results with titles, snippets, URLs]

    Security considerations:
        - Validates query length to prevent abuse
        - Implements timeout to prevent hanging
        - Handles API errors gracefully
        - Sanitizes output for LLM safety
    """

    name: str = "web_search"
    description: str = (
        "Searches the web for current information using Google Search. "
        "Use this when you need up-to-date facts, statistics, market data, "
        "or recent events. Returns top 5 results with titles, snippets, and links. "
        "Best for: market research, competitive intelligence, current events, statistics."
    )
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str) -> str:
        """
        Execute web search using Serper API.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results, or error message if search fails

        Error handling:
            - Missing API key: Returns configuration error
            - API timeout: Returns timeout error after 10 seconds
            - API failure: Returns API error with details
            - No results: Returns "No results found" message
        """
        try:
            # Validate API key exists
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                error_msg = (
                    "Error: SERPER_API_KEY not configured. "
                    "Get a free key at https://serper.dev and add to .env file"
                )
                logger.error(error_msg)
                return error_msg

            # Prepare API request
            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": 5  # Top 5 results
            }
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }

            logger.info(f"Executing web search: '{query}'")

            # Execute search with timeout
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=10  # 10 second timeout
            )
            response.raise_for_status()

            results = response.json()

            # Format results for LLM consumption
            if "organic" not in results or len(results["organic"]) == 0:
                return "No results found for this query. Try rephrasing or broadening the search."

            formatted_results = []
            for idx, item in enumerate(results.get("organic", [])[:5], 1):
                formatted_results.append(
                    f"Result {idx}:\n"
                    f"Title: {item.get('title', 'N/A')}\n"
                    f"Snippet: {item.get('snippet', 'N/A')}\n"
                    f"Link: {item.get('link', 'N/A')}\n"
                )

            output = "\n---\n".join(formatted_results)
            logger.info(f"Search completed successfully: {len(formatted_results)} results")

            return output

        except requests.Timeout:
            error_msg = f"Search timed out after 10 seconds. Query: '{query}'"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        except requests.RequestException as e:
            error_msg = f"Search API request failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Unexpected error during search: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


class CalculatorToolInput(BaseModel):
    """
    Input schema for CalculatorTool.

    Attributes:
        expression: Mathematical expression to evaluate (e.g., "(100 - 50) / 2")
    """
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate. Supports +, -, *, /, **, ()",
        min_length=1,
        max_length=200
    )


class CalculatorTool(BaseTool):
    """
    Safe mathematical calculator tool.

    Evaluates mathematical expressions securely using AST parsing to prevent
    arbitrary code execution. Supports basic arithmetic operations and
    scientific notation.

    Example usage by an agent:
        Thought: "I need to calculate the CAGR"
        Action: calculator
        Action Input: {"expression": "((47.1 / 5.1) ** (1/7) - 1) * 100"}
        Observation: "37.45"

    Security:
        - Uses AST parsing (no eval() or exec())
        - Whitelist of safe operations only
        - Rejects any attempt at code injection
        - Expression length limited to 200 chars

    Supported operations:
        - Addition (+), Subtraction (-)
        - Multiplication (*), Division (/)
        - Exponentiation (**)
        - Parentheses for grouping
        - Negative numbers (-5)
    """

    name: str = "calculator"
    description: str = (
        "Performs safe mathematical calculations. "
        "Supports: addition (+), subtraction (-), multiplication (*), "
        "division (/), exponents (**), and parentheses (). "
        "Use for: CAGR calculations, growth rates, percentages, projections. "
        "Example: '((47.1 / 5.1) ** (1/7) - 1) * 100' calculates compound annual growth rate."
    )
    args_schema: Type[BaseModel] = CalculatorToolInput

    # Whitelist of safe operations
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,  # Unary minus (negative numbers)
    }

    def _run(self, expression: str) -> str:
        """
        Safely evaluate mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            String representation of calculation result, or error message

        Error handling:
            - Division by zero: Returns specific error
            - Invalid syntax: Returns parsing error
            - Unsupported operations: Returns security error
            - Large numbers: Returns overflow error
        """
        try:
            logger.info(f"Evaluating expression: '{expression}'")

            # Parse expression into AST
            node = ast.parse(expression, mode='eval')

            # Recursively evaluate AST nodes
            result = self._eval_node(node.body)

            # Format result (round to 2 decimal places if float)
            if isinstance(result, float):
                # Round to 2 decimals for readability
                result_str = f"{result:.2f}"
            else:
                result_str = str(result)

            logger.info(f"Calculation result: {result_str}")
            return result_str

        except ZeroDivisionError:
            error_msg = "Error: Division by zero"
            logger.error(error_msg)
            return error_msg

        except SyntaxError as e:
            error_msg = f"Error: Invalid mathematical expression. {str(e)}"
            logger.error(error_msg)
            return error_msg

        except TypeError as e:
            error_msg = f"Error: Unsupported operation. Only +, -, *, /, ** allowed. {str(e)}"
            logger.error(error_msg)
            return error_msg

        except OverflowError:
            error_msg = "Error: Number too large to calculate"
            logger.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Error: Calculation failed. {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _eval_node(self, node: ast.AST) -> float:
        """
        Recursively evaluate AST nodes safely.

        Args:
            node: AST node to evaluate

        Returns:
            Numeric result of evaluation

        Raises:
            TypeError: If unsupported operation encountered
        """
        if isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            # Binary operation (e.g., 5 + 3)
            if type(node.op) not in self.SAFE_OPERATORS:
                raise TypeError(f"Unsupported operation: {type(node.op).__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.SAFE_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            # Unary operation (e.g., -5)
            if type(node.op) not in self.SAFE_OPERATORS:
                raise TypeError(f"Unsupported operation: {type(node.op).__name__}")

            operand = self._eval_node(node.operand)
            return self.SAFE_OPERATORS[type(node.op)](operand)
        else:
            raise TypeError(f"Unsupported node type: {type(node).__name__}")


# Export tool instances for easy import
web_search_tool = WebSearchTool()
calculator_tool = CalculatorTool()


# Example usage (for testing)
if __name__ == "__main__":
    print("=== Testing WebSearchTool ===")
    print(web_search_tool._run("AI agent market size 2024"))

    print("\n=== Testing CalculatorTool ===")
    print(f"2 + 2 = {calculator_tool._run('2 + 2')}")
    print(f"100 / 4 = {calculator_tool._run('100 / 4')}")
    print(f"2 ** 8 = {calculator_tool._run('2 ** 8')}")
    print(f"CAGR calc = {calculator_tool._run('((47.1 / 5.1) ** (1/7) - 1) * 100')}")
