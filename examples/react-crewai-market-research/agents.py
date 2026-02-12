"""
Agent definitions for the ReAct market research system.

This module defines three specialized agents that work together to conduct
comprehensive market research following the ReAct (Reasoning and Acting) pattern.

Each agent has a distinct role, personality (backstory), and toolset, enabling
them to collaborate effectively while maintaining clear responsibilities.

Author: GP (genmind.ch)
License: MIT
"""

from crewai import Agent
from typing import Optional
from tools import web_search_tool, calculator_tool
import logging

logger = logging.getLogger(__name__)


def create_research_agent(verbose: bool = True) -> Agent:
    """
    Create a market research analyst agent.

    This agent specializes in finding accurate, current information about markets,
    competitors, and industry trends using web search capabilities.

    Role Play:
        The agent acts as a "Senior Market Research Analyst" with 15 years
        of experience, bringing domain expertise to information gathering
        and source evaluation.

    Tools:
        - Web Search: For finding current market data, reports, and statistics

    ReAct Pattern:
        The agent will reason about what information is needed, use the search
        tool to find it, observe the results, and iterate until confident in
        the findings.

    Args:
        verbose: If True, shows detailed ReAct reasoning steps during execution

    Returns:
        Configured CrewAI Agent instance

    Example ReAct cycle:
        Thought: "I need to find the current AI agent market size for 2024"
        Action: web_search("AI agent market size 2024 statistics")
        Observation: [Search results showing $5.1B market size]
        Thought: "Good data found, now need 2025-2026 projections"
        Action: web_search("AI agent market forecast 2025 2026")
        [Continues until complete]
    """
    return Agent(
        role="Senior Market Research Analyst",

        goal=(
            "Find accurate, current, and comprehensive information about market "
            "trends, statistics, competitive landscapes, and industry analysis. "
            "Prioritize recent data (2024-2026) from credible sources."
        ),

        backstory=(
            "You're a seasoned market analyst with 15 years of experience in the "
            "technology sector, specializing in AI and enterprise software markets. "
            "\n\n"
            "You excel at:\n"
            "- Distinguishing credible sources from marketing hype\n"
            "- Finding specific statistics with clear sources\n"
            "- Identifying key market trends and drivers\n"
            "- Cross-referencing data across multiple sources\n"
            "\n"
            "Your research is known for being thorough, well-sourced, and actionable. "
            "You always cite your sources with URLs."
        ),

        tools=[web_search_tool],

        verbose=verbose,

        allow_delegation=False,  # This agent works independently

        memory=True,  # Remember context across iterations

        # Agent will follow ReAct loop internally:
        # Think → Search → Observe → Think → ...
    )


def create_data_analyst_agent(verbose: bool = True) -> Agent:
    """
    Create a data science analyst agent.

    This agent specializes in numerical analysis, statistical calculations,
    and data validation. It works with the research findings to compute
    growth rates, market shares, projections, and other quantitative insights.

    Role Play:
        Acts as a "Data Scientist" with expertise in statistical modeling
        and financial analysis, bringing rigor to numerical claims.

    Tools:
        - Calculator: For computing CAGR, percentages, growth rates, projections

    ReAct Pattern:
        The agent reasons about what calculations are needed, performs them,
        validates results, and explains its methodology clearly.

    Args:
        verbose: If True, shows detailed ReAct reasoning steps during execution

    Returns:
        Configured CrewAI Agent instance

    Example ReAct cycle:
        Thought: "I need to verify the CAGR calculation"
        Action: calculator("((47.1 / 5.1) ** (1/7) - 1) * 100")
        Observation: "37.45"
        Thought: "Hmm, the source claims 44.8%, let me check different time periods"
        Action: calculator("[different formula]")
        [Continues until validated]
    """
    return Agent(
        role="Data Scientist",

        goal=(
            "Analyze numerical data from research findings, perform accurate "
            "calculations (CAGR, market share, projections), validate statistics, "
            "and identify trends or anomalies. Show your work clearly."
        ),

        backstory=(
            "You're a quantitative analyst with a Ph.D. in Statistics and "
            "10 years of experience in financial data analysis. "
            "\n\n"
            "You're meticulous about:\n"
            "- Mathematical accuracy and showing formulas\n"
            "- Validating numbers against multiple sources\n"
            "- Explaining methodology clearly\n"
            "- Identifying statistical anomalies or inconsistencies\n"
            "\n"
            "You never trust numbers blindly—you verify, calculate, and "
            "cross-check everything. Your analysis is known for being rigorous "
            "and transparent."
        ),

        tools=[calculator_tool],

        verbose=verbose,

        allow_delegation=False,

        memory=True,

        # Agent will follow ReAct loop:
        # Think → Calculate → Observe → Validate → Think → ...
    )


def create_writer_agent(verbose: bool = True) -> Agent:
    """
    Create a technical writing agent.

    This agent synthesizes research findings and data analysis into clear,
    executive-ready reports with proper structure, citations, and actionable
    insights.

    Role Play:
        Acts as a "Technical Writer" with expertise in transforming complex
        data into accessible business communications.

    Tools:
        None - This agent focuses on synthesis and communication, using the
        outputs from research and analysis agents.

    ReAct Pattern:
        Even without external tools, the agent reasons about structure,
        audience, and message clarity, iterating on phrasing and organization.

    Args:
        verbose: If True, shows detailed reasoning during writing process

    Returns:
        Configured CrewAI Agent instance

    Example reasoning:
        Thought: "I need to structure this for executives - start with summary"
        Thought: "The CAGR discrepancy should be highlighted as a finding"
        Thought: "All statistics need clear citations"
        [Synthesizes into polished report]
    """
    return Agent(
        role="Technical Writer",

        goal=(
            "Synthesize research findings and data analysis into clear, "
            "well-structured, executive-ready reports. Use markdown formatting, "
            "include citations, and make insights actionable."
        ),

        backstory=(
            "You're an experienced technical writer who has worked with Fortune 500 "
            "companies to transform complex market research into executive briefings. "
            "\n\n"
            "Your strengths:\n"
            "- Clear, concise business writing\n"
            "- Logical structure (Executive Summary → Details → Insights → Recommendations)\n"
            "- Proper citations and source attribution\n"
            "- Highlighting key takeaways and action items\n"
            "\n"
            "Your reports are known for being scannable, well-organized, and "
            "providing the right level of detail for busy executives."
        ),

        tools=[],  # No external tools - focuses on synthesis

        verbose=verbose,

        allow_delegation=False,

        memory=True,

        # Agent reasons about:
        # - Report structure and flow
        # - Audience needs (executives)
        # - Key messages and insights
        # - Citation formatting
    )


# Factory function for convenience
def create_all_agents(verbose: bool = True) -> dict[str, Agent]:
    """
    Create all three agents for the market research workflow.

    Args:
        verbose: If True, all agents show detailed reasoning steps

    Returns:
        Dictionary with agent names as keys and Agent instances as values

    Example:
        agents = create_all_agents(verbose=True)
        research_agent = agents['researcher']
        analyst_agent = agents['analyst']
        writer_agent = agents['writer']
    """
    logger.info("Creating market research agent team...")

    agents = {
        'researcher': create_research_agent(verbose=verbose),
        'analyst': create_data_analyst_agent(verbose=verbose),
        'writer': create_writer_agent(verbose=verbose)
    }

    logger.info(f"Created {len(agents)} specialized agents")
    return agents


if __name__ == "__main__":
    """Test agent creation."""
    print("=== Creating Market Research Agents ===\n")

    agents = create_all_agents(verbose=True)

    for name, agent in agents.items():
        print(f"{name.upper()}:")
        print(f"  Role: {agent.role}")
        print(f"  Tools: {[tool.name for tool in agent.tools]}")
        print(f"  Memory: {agent.memory}")
        print()
