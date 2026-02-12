"""
Task definitions for the ReAct market research workflow.

Tasks define WHAT needs to be done, while Agents define WHO does it.
Each task includes clear instructions, expected output format, and is
assigned to a specific agent.

The three-phase workflow:
    1. Research Phase: Gather market data and sources
    2. Analysis Phase: Validate numbers and compute metrics
    3. Writing Phase: Synthesize into executive report

Author: GP (genmind.ch)
License: MIT
"""

from crewai import Task, Agent
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_research_task(
    topic: str,
    agent: Agent,
    context: Optional[str] = None
) -> Task:
    """
    Create a market research task.

    This task instructs the research agent to gather comprehensive market
    data, statistics, and competitive intelligence on the given topic.

    The agent will use the ReAct pattern to:
        - Reason about what information is needed
        - Search for credible sources
        - Observe and evaluate results
        - Iterate until comprehensive coverage achieved

    Args:
        topic: Research topic (e.g., "AI agent market size 2024-2026")
        agent: Research agent instance to execute this task
        context: Optional additional context or constraints

    Returns:
        Configured CrewAI Task instance

    Example output:
        # Research Summary: AI Agent Market Size 2024-2026

        ## Market Size Data
        - 2024: $5.1 billion (Source: [URL])
        - 2025: $10.2 billion projected (Source: [URL])
        - 2026: $18.5 billion projected (Source: [URL])

        ## Key Players
        - Microsoft: Azure AI Agent Service
        - Google: Vertex AI Agents
        [...]

        ## Growth Drivers
        1. Enterprise automation demand
        2. Conversational AI adoption
        [...]
    """
    description = f"""Research the following topic thoroughly: {topic}

Your research should cover:

1. **Market Size and Growth** (2024-2026)
   - Current market size with specific dollar figures
   - Historical growth data (if available)
   - Projected growth rates and forecasts
   - Source every statistic with URL

2. **Competitive Landscape**
   - Key players and market share data
   - Major vendors and their offerings
   - Recent funding rounds or acquisitions
   - Competitive positioning

3. **Market Trends and Drivers**
   - 3-5 major trends shaping the market
   - Technology developments
   - Regulatory or compliance factors
   - Industry adoption patterns

4. **Use Cases and Applications**
   - Primary use cases (customer service, IT ops, etc.)
   - Industry-specific applications
   - Success stories or case studies

**Critical Instructions**:
- Prioritize recent data (2024-2026)
- Every statistic MUST include source URL
- Look for credible sources (Gartner, IDC, McKinsey, industry reports)
- If data conflicts across sources, note the discrepancy
- Focus on facts, not marketing claims

{f'**Additional Context**: {context}' if context else ''}
"""

    expected_output = """A comprehensive research summary with:

1. **Market Size Section**
   - Specific dollar figures for 2024-2026
   - Growth rates (YoY, CAGR)
   - All claims cited with URLs

2. **Competitive Analysis**
   - Top 5-10 key players
   - Market share data (if available)
   - Product/service offerings

3. **Trends Section**
   - 3-5 major trends with supporting evidence
   - Each trend explained in 2-3 sentences

4. **Use Cases**
   - Primary applications and verticals
   - Real-world examples

**Format**: Structured markdown with clear headers and bullet points
**Length**: 500-800 words
**Citations**: Every factual claim must have source URL in format [Source](URL)
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def create_analysis_task(
    research_output: str,
    agent: Agent
) -> Task:
    """
    Create a data analysis task.

    This task instructs the data analyst agent to validate, compute, and
    analyze the numerical data from the research findings.

    The agent will use the ReAct pattern to:
        - Reason about what calculations are needed
        - Perform calculations using the calculator tool
        - Observe results and validate against sources
        - Identify anomalies or discrepancies

    Args:
        research_output: Output from the research task (market data and statistics)
        agent: Data analyst agent instance to execute this task

    Returns:
        Configured CrewAI Task instance

    Example output:
        # Data Analysis Report

        ## CAGR Calculation
        Formula: ((End Value / Start Value) ^ (1 / Years)) - 1
        Calculation: ((47.1 / 5.1) ** (1/7) - 1) * 100
        Result: 37.45% CAGR

        Note: Source claims 44.8% CAGR - discrepancy may be due to...

        ## Market Share Analysis
        - Microsoft: 35% (calculated from revenue figures)
        - Google: 28%
        [...]
    """
    description = f"""Analyze the numerical data from the research findings below:

{research_output}

Perform the following analyses:

1. **CAGR Calculations**
   - Calculate compound annual growth rate for market size
   - Verify against any CAGR claims in sources
   - Show your formula and calculation steps
   - Note any discrepancies

2. **Market Share Analysis**
   - If revenue/size data available for vendors, calculate market shares
   - Express as percentages
   - Validate that percentages sum to reasonable total (<100%)

3. **Growth Projections**
   - Based on historical data and stated CAGR, project future values
   - Calculate year-over-year growth rates
   - Identify acceleration or deceleration trends

4. **Statistical Validation**
   - Check if numbers are consistent across claims
   - Identify outliers or unlikely figures
   - Note confidence level in each statistic

**Critical Instructions**:
- SHOW YOUR WORK: Include formulas and calculation steps
- Use the calculator tool for all computations
- If sources conflict, recalculate to verify which is correct
- State assumptions clearly (e.g., "Assuming linear growth...")
- Highlight any data quality issues
"""

    expected_output = """A quantitative analysis report with:

1. **CAGR Calculations**
   - Formulas shown
   - Step-by-step calculations
   - Verification against source claims
   - Any discrepancies noted

2. **Market Share Breakdown**
   - Vendor percentages with basis for calculation
   - Total market validation

3. **Projections**
   - 2025, 2026 market size projections
   - Growth rate trends
   - Confidence levels (high/medium/low)

4. **Data Quality Assessment**
   - Which statistics are most reliable
   - Where data is missing or unclear
   - Recommended caveats for final report

**Format**: Structured markdown with calculations clearly shown
**Style**: Technical but clear - explain methodology
**Length**: 300-500 words
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def create_writing_task(
    research_output: str,
    analysis_output: str,
    agent: Agent,
    topic: str
) -> Task:
    """
    Create a report writing task.

    This task instructs the writer agent to synthesize the research and
    analysis into a polished, executive-ready report.

    The agent will reason about:
        - Optimal structure for the audience
        - Key insights and takeaways
        - Proper citation formatting
        - Actionable recommendations

    Args:
        research_output: Output from research task (market data)
        analysis_output: Output from analysis task (validated numbers)
        agent: Writer agent instance to execute this task
        topic: Original research topic for report title

    Returns:
        Configured CrewAI Task instance

    Example output:
        # Market Research Report: AI Agent Market Size 2024-2026

        ## Executive Summary
        The AI agents market is experiencing explosive growth...
        [3-4 sentences with key findings]

        ## Market Overview
        [Detailed market size, players, trends]

        ## Key Findings
        1. Market growing at 37% CAGR
        2. Enterprise adoption accelerating
        3. [...]

        ## Recommendations
        - [Actionable insights]
    """
    description = f"""Write a comprehensive market analysis report based on the research and analysis below.

**RESEARCH FINDINGS**:
{research_output}

**DATA ANALYSIS**:
{analysis_output}

Structure your report as follows:

1. **Executive Summary** (3-4 sentences)
   - Market size and growth rate
   - Top 2-3 key findings
   - Investment/opportunity assessment

2. **Market Overview**
   - Current market size (2024) with source
   - Projected growth (2024-2026)
   - CAGR and growth drivers

3. **Competitive Landscape**
   - Key players and market positioning
   - Market share data (if available)
   - Recent developments

4. **Market Trends**
   - 3-5 major trends
   - Each with supporting data and sources
   - Impact on market growth

5. **Key Findings** (Bullet Points)
   - 4-6 critical insights
   - Data-driven and specific
   - Actionable for decision-makers

6. **Recommendations**
   - 2-3 strategic recommendations
   - Based on data and trends
   - Specific and actionable

**Critical Instructions**:
- Write for busy executives (scannable, concise)
- CITE ALL SOURCES: Use markdown links [Source Name](URL)
- Use markdown formatting: headers, bold, bullet points
- Highlight numbers and percentages in **bold**
- If data has caveats (e.g., conflicting sources), note them
- Professional tone, clear language

**Title**: Market Research Report: {topic}
"""

    expected_output = """A polished market analysis report with:

**Structure**:
- Title (# heading)
- Executive Summary
- Market Overview
- Competitive Landscape
- Market Trends
- Key Findings (bullet points)
- Recommendations

**Quality Standards**:
- Executive-ready language
- All statistics cited with markdown links
- Clear section headers (##)
- Scannable format (bullets, bold numbers)
- 700-1000 words
- Proper markdown formatting

**Style**:
- Professional but accessible
- Data-driven insights
- Specific, actionable recommendations
- No marketing fluff
"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


if __name__ == "__main__":
    """Test task creation."""
    from agents import create_all_agents

    print("=== Testing Task Creation ===\n")

    agents = create_all_agents(verbose=False)
    topic = "AI agent market size 2024-2026"

    # Create tasks
    research_task = create_research_task(topic, agents['researcher'])
    print(f"Research Task Created")
    print(f"  Description length: {len(research_task.description)} chars")
    print(f"  Agent role: {research_task.agent.role}\n")

    # For analysis and writing, we'd need actual research output
    # This is just structural testing
    print("Analysis and Writing tasks would follow with actual data...")
