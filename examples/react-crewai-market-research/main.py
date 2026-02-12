#!/usr/bin/env python3
"""
ReAct Pattern Market Research Assistant - Main Orchestration

This script demonstrates a production-ready implementation of the ReAct
(Reasoning and Acting) pattern using CrewAI for market research automation.

The system uses three specialized agents that follow the ReAct loop:
    1. Research Agent → Gathers market data using web search
    2. Data Analyst Agent → Validates and computes metrics
    3. Writer Agent → Synthesizes into executive report

Usage:
    python main.py "AI agent market size 2024-2026"
    python main.py "blockchain adoption in banking" --verbose
    python main.py "quantum computing market trends" --output my-report.md

Author: GP (genmind.ch)
License: MIT
Blog Post: https://genmind.ch/posts/Building-ReAct-Agents-with-CrewAI-From-Theory-to-Production/
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# CrewAI imports
from crewai import Crew, Process

# Local imports
from agents import create_research_agent, create_data_analyst_agent, create_writer_agent
from tasks import create_research_task, create_analysis_task, create_writing_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketResearchCrew:
    """
    Orchestrates a multi-agent market research workflow using the ReAct pattern.

    This class manages the three-phase research process:
        Phase 1: Research → Gather market data
        Phase 2: Analysis → Validate numbers and compute metrics
        Phase 3: Writing → Synthesize into report

    Each phase is executed sequentially, with outputs feeding into the next phase.
    """

    def __init__(self, topic: str, verbose: bool = True):
        """
        Initialize the market research crew.

        Args:
            topic: Research topic (e.g., "AI agent market size 2024-2026")
            verbose: If True, shows detailed ReAct reasoning steps
        """
        self.topic = topic
        self.verbose = verbose
        self.start_time = datetime.now()

        logger.info(f"Initializing MarketResearchCrew for topic: '{topic}'")

        # Create specialized agents
        self.research_agent = create_research_agent(verbose=verbose)
        self.analyst_agent = create_data_analyst_agent(verbose=verbose)
        self.writer_agent = create_writer_agent(verbose=verbose)

        logger.info("All agents created successfully")

    def execute(self) -> str:
        """
        Execute the three-phase market research workflow.

        Returns:
            Final market research report as markdown string

        Raises:
            RuntimeError: If any phase fails
        """
        try:
            # Phase 1: Research
            logger.info("\n" + "="*60)
            logger.info("🔍 PHASE 1: MARKET RESEARCH")
            logger.info("="*60)

            research_result = self._execute_research_phase()

            # Phase 2: Data Analysis
            logger.info("\n" + "="*60)
            logger.info("📊 PHASE 2: DATA ANALYSIS")
            logger.info("="*60)

            analysis_result = self._execute_analysis_phase(research_result)

            # Phase 3: Report Writing
            logger.info("\n" + "="*60)
            logger.info("✍️  PHASE 3: REPORT WRITING")
            logger.info("="*60)

            final_report = self._execute_writing_phase(research_result, analysis_result)

            # Calculate execution time
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info("\n" + "="*60)
            logger.info(f"✅ MARKET RESEARCH COMPLETE")
            logger.info(f"   Total execution time: {duration:.1f} seconds")
            logger.info("="*60 + "\n")

            return final_report

        except Exception as e:
            logger.error(f"Market research failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Market research execution failed: {str(e)}")

    def _execute_research_phase(self) -> str:
        """
        Execute Phase 1: Market Research.

        Creates a crew with the research agent and task, executes it,
        and returns the research findings.

        Returns:
            Research summary with market data and sources
        """
        research_task = create_research_task(
            topic=self.topic,
            agent=self.research_agent
        )

        research_crew = Crew(
            agents=[self.research_agent],
            tasks=[research_task],
            process=Process.sequential,
            verbose=self.verbose
        )

        result = research_crew.kickoff()
        logger.info("Research phase completed")

        return str(result)

    def _execute_analysis_phase(self, research_output: str) -> str:
        """
        Execute Phase 2: Data Analysis.

        Takes research findings, creates analysis task, and executes it
        with the data analyst agent.

        Args:
            research_output: Output from research phase

        Returns:
            Analysis report with validated statistics and calculations
        """
        analysis_task = create_analysis_task(
            research_output=research_output,
            agent=self.analyst_agent
        )

        analysis_crew = Crew(
            agents=[self.analyst_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=self.verbose
        )

        result = analysis_crew.kickoff()
        logger.info("Analysis phase completed")

        return str(result)

    def _execute_writing_phase(
        self,
        research_output: str,
        analysis_output: str
    ) -> str:
        """
        Execute Phase 3: Report Writing.

        Takes research and analysis outputs, synthesizes them into
        a polished executive-ready report.

        Args:
            research_output: Output from research phase
            analysis_output: Output from analysis phase

        Returns:
            Final markdown report
        """
        writing_task = create_writing_task(
            research_output=research_output,
            analysis_output=analysis_output,
            agent=self.writer_agent,
            topic=self.topic
        )

        writing_crew = Crew(
            agents=[self.writer_agent],
            tasks=[writing_task],
            process=Process.sequential,
            verbose=self.verbose
        )

        result = writing_crew.kickoff()
        logger.info("Writing phase completed")

        return str(result)


def save_report(content: str, output_file: str) -> None:
    """
    Save market research report to file.

    Args:
        content: Report content (markdown)
        output_file: Output file path

    Raises:
        IOError: If file write fails
    """
    try:
        output_path = Path(output_file)

        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Get file size
        size = output_path.stat().st_size

        logger.info(f"📝 Report saved to: {output_path}")
        logger.info(f"   File size: {size:,} bytes")

    except Exception as e:
        logger.error(f"Failed to save report: {str(e)}")
        raise IOError(f"Could not write to {output_file}: {str(e)}")


def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate required environment variables are set.

    Returns:
        Tuple of (is_valid, list_of_missing_vars)
    """
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for LLM',
        'SERPER_API_KEY': 'Serper API key for web search'
    }

    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")

    return len(missing) == 0, missing


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='ReAct Pattern Market Research Assistant',
        epilog='Example: python main.py "AI agent market size 2024-2026"'
    )
    parser.add_argument(
        'topic',
        type=str,
        help='Research topic to analyze'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show detailed ReAct reasoning steps (default: True)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='market_research_report.md',
        help='Output file path (default: market_research_report.md)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override LLM model (default: from environment or gpt-4o)'
    )

    args = parser.parse_args()

    # Validate environment
    is_valid, missing_vars = validate_environment()
    if not is_valid:
        logger.error("❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        logger.error("\nPlease configure your .env file (see .env.example)")
        sys.exit(1)

    # Override model if specified
    if args.model:
        os.environ['OPENAI_MODEL'] = args.model
        logger.info(f"Using LLM model: {args.model}")

    # Print startup banner
    print("\n" + "="*60)
    print("🚀 ReAct Pattern Market Research Assistant")
    print("="*60)
    print(f"📋 Topic: {args.topic}")
    print(f"📝 Output: {args.output}")
    print(f"🔊 Verbose: {args.verbose}")
    print("="*60 + "\n")

    try:
        # Create and execute crew
        crew = MarketResearchCrew(
            topic=args.topic,
            verbose=args.verbose
        )

        result = crew.execute()

        # Save report
        save_report(result, args.output)

        # Print success message
        print("\n" + "="*60)
        print("✅ SUCCESS")
        print("="*60)
        print(f"Report saved to: {args.output}")
        print("\nNext steps:")
        print(f"  - Review the report: cat {args.output}")
        print("  - Open in your editor for refinement")
        print("  - Share with stakeholders")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Research interrupted by user (Ctrl+C)")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}")
        logger.error("\nTroubleshooting tips:")
        logger.error("  1. Check your API keys in .env file")
        logger.error("  2. Verify you have internet connectivity")
        logger.error("  3. Check OpenAI API rate limits")
        logger.error("  4. Try a simpler/shorter research topic")
        sys.exit(1)


if __name__ == "__main__":
    main()
