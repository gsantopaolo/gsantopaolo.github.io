#!/usr/bin/env python3
"""
ReAct Pattern Market Research Assistant - Main Entry Point

Demonstrates the ReAct (Reasoning and Acting) pattern using Microsoft Agent Framework.

The ReAct pattern interleaves reasoning and action in a continuous loop:
Think → Act → Observe → Think → Act → ...

This contrasts with Planning which creates a strategy upfront, then executes.

Usage:
    python main.py "AI agent market size 2024-2026"
    python main.py "quantum computing enterprise adoption" --output quantum_report.md
    python main.py "autonomous vehicles market forecast" --streaming

Author: GP (genmind.ch)
License: MIT
Blog Post: https://genmind.ch/posts/Building-ReAct-Agents-with-Microsoft-Agent-Framework-From-Theory-to-Production/
"""

import sys
import os
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Awaitable

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from agent_framework import Agent, FunctionInvocationContext
from agent_framework.azure import AzureOpenAIChatClient

from tools import web_search, calculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReActMarketResearchAgent:
    """
    ReAct Pattern agent for market research automation.

    This agent demonstrates the ReAct pattern where reasoning and action
    are continuously interleaved:
        1. THINK: Agent reasons about what information it needs
        2. ACT: Agent uses tools (web_search, calculator) to get data
        3. OBSERVE: Agent analyzes tool outputs
        4. REPEAT: Loop continues until task is complete

    Contrast with Planning pattern:
        - ReAct: Continuous loop, adapts dynamically
        - Planning: Create plan upfront, execute sequentially
    """

    def __init__(self, topic: str, enable_logging: bool = True):
        """
        Initialize the ReAct market research agent.

        Args:
            topic: Research topic (e.g., "AI agent market size 2024-2026")
            enable_logging: If True, logs ReAct iterations (THINK-ACT-OBSERVE)
        """
        self.topic = topic
        self.enable_logging = enable_logging
        self.start_time = datetime.now()

        logger.info(f"Initializing ReAct agent for topic: '{topic}'")

        # Configure Azure OpenAI client
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

        if not all([api_key, endpoint, deployment_name]):
            raise ValueError(
                "Missing required environment variables. Please set:\n"
                "  - AZURE_OPENAI_API_KEY\n"
                "  - AZURE_OPENAI_ENDPOINT\n"
                "  - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME\n"
                "See .env.example for details."
            )

        self.client = AzureOpenAIChatClient(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name
        )

        # Create ReAct agent
        self.agent = self._create_agent()

        logger.info("ReAct agent created successfully")

    def _create_agent(self) -> Agent:
        """
        Create the ReAct agent with tools and instructions.

        Returns:
            Configured Agent with ReAct capabilities
        """
        agent = Agent(
            client=self.client,
            name="market_research_agent",
            instructions=f"""You are a Market Research AI Agent using the ReAct pattern.

For each task, you MUST follow the ReAct loop:
1. THINK: Reason about what information you need
2. ACT: Use tools (web_search, calculator) to gather data
3. OBSERVE: Analyze the tool outputs
4. REPEAT: Continue until you have complete information

Your goal: Research: {self.topic}

Provide a comprehensive market research report including:
- Current market size with sources
- Growth projections and CAGR calculations
- Key market drivers and trends
- Competitive landscape
- Recommendations

CRITICAL REQUIREMENTS:
- Always cite sources with URLs
- Show calculations explicitly
- Verify numbers from multiple sources
- Use web_search for facts, calculator for metrics
- Be thorough but concise""",
            tools=[web_search, calculator]
        )

        # Add logging middleware to observe ReAct iterations
        if self.enable_logging:
            agent.add_function_middleware(self._logging_middleware)

        return agent

    async def _logging_middleware(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        """
        Middleware to log each ReAct iteration.

        Logs:
        - THINK → ACT: What tool the agent decided to use
        - ACT: Tool execution
        - OBSERVE: Tool results
        """
        # THINK → ACT: Log before function execution
        logger.info(f"\n🤔 THINK → ACT: Calling {context.function.name}")
        logger.info(f"   Arguments: {context.arguments}")

        # Execute tool (ACT phase)
        await call_next(context)

        # OBSERVE: Log after function execution
        result_preview = str(context.result)[:200] if context.result else "None"
        logger.info(f"✅ OBSERVE: {context.function.name} completed")
        logger.info(f"   Result preview: {result_preview}...")

    async def execute(self) -> str:
        """
        Execute the ReAct workflow.

        Returns:
            Final market research report as string

        Raises:
            RuntimeError: If execution fails
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("🔍 REACT PATTERN AGENT - START")
            logger.info("=" * 60)
            logger.info(f"Research Topic: {self.topic}\n")

            # Create thread for multi-turn conversation
            thread = self.agent.get_new_thread()

            # Run agent (executes ReAct loop automatically)
            logger.info("🔄 Starting ReAct loop (Think → Act → Observe → repeat)...")

            result = await self.agent.run(
                f"Research the following topic and provide a comprehensive report: {self.topic}",
                thread=thread
            )

            # Extract final report from result
            if result and result.messages:
                final_message = result.messages[-1]
                if final_message.contents:
                    report = final_message.contents[0].text

                    # Calculate execution time
                    duration = (datetime.now() - self.start_time).total_seconds()

                    logger.info("\n" + "=" * 60)
                    logger.info("✅ REACT WORKFLOW COMPLETE")
                    logger.info(f"   Total execution time: {duration:.1f} seconds")
                    logger.info("=" * 60 + "\n")

                    return report
                else:
                    raise RuntimeError("Agent produced no output content")
            else:
                raise RuntimeError("Agent produced no messages")

        except Exception as e:
            logger.error(f"ReAct agent execution failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"ReAct agent failed: {str(e)}")

    async def execute_streaming(self):
        """
        Execute the ReAct workflow with streaming responses.

        Streams incremental responses as the agent iterates through
        the ReAct loop, providing real-time feedback.
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("🔍 REACT PATTERN AGENT - STREAMING MODE")
            logger.info("=" * 60)
            logger.info(f"Research Topic: {self.topic}\n")

            # Create thread
            thread = self.agent.get_new_thread()

            # Stream responses
            async for chunk in self.agent.run_stream(
                f"Research the following topic and provide a comprehensive report: {self.topic}",
                thread=thread
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)

            print("\n")

            # Calculate execution time
            duration = (datetime.now() - self.start_time).total_seconds()

            logger.info("\n" + "=" * 60)
            logger.info("✅ REACT WORKFLOW COMPLETE (STREAMING)")
            logger.info(f"   Total execution time: {duration:.1f} seconds")
            logger.info("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"ReAct agent streaming failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"ReAct agent streaming failed: {str(e)}")


def save_report(content: str, output_file: str) -> None:
    """
    Save market research report to file.

    Args:
        content: Report content
        output_file: Output file path

    Raises:
        IOError: If file write fails
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        size = output_path.stat().st_size

        logger.info(f"📝 Report saved to: {output_path}")
        logger.info(f"   File size: {size:,} bytes")

    except Exception as e:
        logger.error(f"Failed to save report: {str(e)}")
        raise IOError(f"Could not write to {output_file}: {str(e)}")


async def async_main(args):
    """Async main function."""
    start_time = datetime.now()

    try:
        # Create ReAct agent
        agent = ReActMarketResearchAgent(
            topic=args.topic,
            enable_logging=args.verbose
        )

        # Execute workflow (streaming or regular)
        if args.streaming:
            await agent.execute_streaming()
            # Note: In streaming mode, output goes to stdout
        else:
            result = await agent.execute()

            # Save report
            if not args.streaming:
                save_report(result, args.output)

        # Calculate total time
        duration = (datetime.now() - start_time).total_seconds()

        # Print success message
        if not args.streaming:
            print("\n" + "=" * 60)
            print("✅ SUCCESS")
            print("=" * 60)
            print(f"Report saved to: {args.output}")
            print(f"Execution time: {duration:.1f} seconds")
            print("\nNext steps:")
            print(f"  - Review the report: cat {args.output}")
            print("  - Open in your editor for refinement")
            print("  - Share with stakeholders")
            print("=" * 60 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Research interrupted by user (Ctrl+C)")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}")
        logger.error("\nTroubleshooting tips:")
        logger.error("  1. Check your API keys in .env file")
        logger.error("  2. Verify internet connectivity")
        logger.error("  3. Check Azure OpenAI API status")
        logger.error("  4. Try a simpler research topic")
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='ReAct Pattern Market Research Assistant (Microsoft Agent Framework)',
        epilog='Example: python main.py "AI agent market size 2024-2026"',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'topic',
        type=str,
        help='Research topic to analyze'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='react_market_report.md',
        help='Output file path (default: react_market_report.md)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show detailed ReAct reasoning steps (default: True)'
    )

    parser.add_argument(
        '--streaming',
        action='store_true',
        default=False,
        help='Enable streaming mode for real-time output'
    )

    args = parser.parse_args()

    # Print startup banner
    print("\n" + "=" * 60)
    print("🔄 ReAct Pattern Market Research Assistant")
    print("   Framework: Microsoft Agent Framework")
    print("=" * 60)
    print(f"📋 Topic: {args.topic}")
    if not args.streaming:
        print(f"📝 Output: {args.output}")
    print(f"🔊 Verbose: {args.verbose}")
    print(f"📡 Streaming: {args.streaming}")
    print("=" * 60 + "\n")

    # Run async main
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
