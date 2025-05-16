---
title: 'CV-Pilot: A Hands-On Agentic Experiment'
date: '2025-04-05T10:31:30+00:00'
author: gp
layout: post
image: /content/2025/04/cv-pilot.png
categories: [Agents, LLM, CrewAI]
---


## CV-Pilot: A Hands-On Agentic Experiment

At [genmind.ch](https://genmind.ch/), I'm always on the lookout for ways to push the 
boundaries of what GenAI can do for productivity.

Here’s why I believe agentic AI is the next big leap, what its strengths and challenges are, 
and how everyone from startups to Salesforce is racing to build with it




## Why Agentic AI?

Agentic AI frameworks—Microsoft AutoGen, Semantic Kernel, LangChain, CrewAI, 
and more—provide structured, goal-driven agents that coordinate to solve complex tasks. 
Unlike single-prompt chatbots, these systems:

* Define **specialized roles** (data collector, analyzer, editor).
* Chain together **multi-step workflows** with memory and tools.
* Offer **traceability** and checkpoints for human oversight.

I’m extending Crew AI’s sample code into a “CV-Pilot” app, wiring up real scraping tools, 
file readers, and audit trails—so you can see exactly how each agent arrives at its output.

## Pros and Cons of Agentic AI

**Pros**

* **Autonomous Reasoning & Planning**: Agents can self-check, plan sub-tasks, and adapt 
to dynamic goals, closing skill gaps in talent-scarce industries.
* **Efficiency & Focus**: Routine, multi-step workflows—expense approvals, data extraction, 
onboarding—are automated, freeing humans for strategic decision-making.
* **Proactive Initiative**: Modern agents anticipate needs, flag anomalies, or suggest next 
steps without explicit prompts).
* **Security Automation**: In cybersecurity and support, agents can preempt incidents and 
handle high volumes of queries at scale.

**Cons**

* **Job Displacement**: Automating complex tasks risks replacing roles, potentially 
increasing unemployment and inequality.
* **Ethical & Bias Risks**: Training data biases can propagate unfair decisions, 
and opaque decision paths complicate accountability.
* **High Costs & Complexity**: Upfront investments in infrastructure, integration, 
and specialized talent can be prohibitive for smaller organizations.
* **Security & Privacy Vulnerabilities**: Autonomous data access amplifies the attack surface, 
raising breach and misuse concerns.
* **Environmental Impact**: Running large-scale, always-on agents consumes significant 
compute resources, heightening sustainability challenges.

## Why Everything Is Going Agentic

The shift from static automation to **dynamic, goal-oriented agents** mirrors demands 
for adaptable, resilient systems. Agents don’t just execute code—they reason about 
objectives, recover from failures, and replan on the fly. As **Harvard Business Review** 
notes, we’re already seeing agents that can plan travel, manage supply chains, or act as 
virtual caregivers—ushering in a new era of human-AI 
collaboration. In finance and HR, AI agents are autonomously handling expense approvals 
and onboarding processes, boosting productivity without ballooning headcounts.

## Big Companies Embracing Agentic AI

It’s not just startups riding the hype—**Salesforce** has rebranded its AI initiatives as 
“Agentforce,” deploying agents that manage over 80% of routine customer queries, cutting 
costs and improving satisfaction. Meanwhile, **Andreessen Horowitz** is pouring billions 
into agent-focused ventures, even as some warn of a marketing oversell. And under the hood, 
Microsoft and OpenAI continue to advance frameworks—AutoGen, Semantic Kernel, 
GPT-powered tools—cementing agentic architecture as the backbone of next-gen enterprise AI.

## Looking Ahead: The Future of AI Agents

As foundational LLMs grow ever more capable, agents will not only execute tasks but 
**learn and optimize** over time, forming continuous improvement loops. 
The **World Economic Forum** predicts agents will shoulder specialized coding assignments, 
real-time analytics, and more—closing critical skill gaps in fast-moving sectors. 
With advances in model efficiency, multimodal reasoning, and on-device inference, the next 
wave of AI agents promises autonomy and adaptability far beyond today’s demos. Building off 
Crew AI’s flexible, role-based core, I’m excited to see how powerful agents will reshape 
workflows—and what new possibilities emerge as we hit higher LLM performance thresholds.

## Why Crew AI Agentic Workflows?

Crew AI provides a flexible “multi‐agent” framework that lets you:

1. **Define agents** with distinct roles and toolsets.
2. **Compose tasks** that map to concrete, testable steps.
3. **Orchestrate** everything under a manager agent with built-in planning, delegation, 
and traceability.

In a world where job applications are repetitive yet demand precision, this architecture
seemed tailor-made to automate both CV customization and cover-letter drafting—without 
losing a human-in-the-loop checkpoint.



## Introducing CV-Pilot

To test agentic by myself, I took the official Crew AI sample and built
[**CV-Pilot**](https://github.com/gsantopaolo/cv-pilot)—an 
AI-powered toolkit that automates resume tailoring and motivation-letter writing through 
two independent pipelines, 
**gen\_resume** and **gen\_motivation**. Here’s how I extended the sample, 
put its agentic orchestration to the test, and why the results are already looking very 
promising.

---


CV-Pilot consists of two CLI pipelines:

* **gen\_resume**

  * **Input**: raw `cv_.md` + job posting URL or Markdown
  * **Output**: `docs/new_resume.md`
  * **Flow**: extract requirements → profile match → tailored resume

* **gen\_motivation**

  * **Input**: `new_resume.md` + `company_url` + job posting
  * **Output**: `docs/motivation_letter.md`
  * **Flow**: scrape company site → parse job ad → draft intro/body/conclusion

Each pipeline dumps a JSON audit trail (`application_state.json`) so you can see every prompt, 
source URL, and intermediate draft. 
Human review tags (`<!-- REVIEW: … -->`) ensure you stay in control.

---

### Extending the Sample: What I Changed

1. **Configuring Agents and Tasks**
   I forked the Crew AI sample, then defined two new crews—`JobApplicationCrew` for resumes 
and `MotivationLetterCrew` for cover letters—each wired to my own `config/agents.yaml` 
and `config/tasks.yaml`. That let me specify different back-stories, goals, and tool sets 
per agent.

2. **Custom Tools**

   * **SerperDevTool** for web search
   * **ScrapeWebsiteTool** for company info
   * **DirectoryReadTool**, **FileReadTool**, **FileWriterTool** for local docs
     All tools use simple caching to avoid redundant calls during development.

3. **Human-in-the-Loop**
   After each draft pipeline, execution pauses. I review `new_resume.md` or 
`motivation_letter.md`, provide feedback in Markdown comments, then resume. 
This keeps the AI from “going off rails.”

4. **LangTrace Integration**
   By plugging in the LangTrace SDK, I get spans around each agentic step, so I can 
visualize the end-to-end timeline in my observability dashboard.

---

### Testing the Agentic Orchestration

With everything wired up, I ran a real job posting through both pipelines:

```bash
# 1. Tailor resume
python3 gen_resume.py --doc_path ./docs \
    --job_posting_url "https://jobs.example.com/123"

# (Review and adjust docs/new_resume.md)

# 2. Generate cover letter
python3 gen_motivation.py \
    --company_url "https://example.com" \
    --job_posting_url "https://jobs.example.com/123" \
    --doc_path ./docs
```

**Key observations:**

* **Relevance:** The tailored resume highlights exactly the skills and keywords required in 
the posting.
* **Coherence:** The motivation letter follows a clear intro–body–conclusion structure, 
citing mission statements and company values pulled directly from the website.
* **Speed:** Both pipelines run in under a minute, even with multiple web scrapes.
* **Traceability:** The JSON state file logs every decision—ideal for audits or fine-tuning 
prompts.

---

### Results & Next Steps

I’m impressed by how seamlessly the agents collaborated:

* The **manager** agent orchestrated extraction, profiling, and drafting without manual 
handoffs.
* The **editor** agent produced polished output that only needed light human tweaks.
* Switching the underlying LLM (OpenAI, Anthropic, Gemini) was as simple as changing an
ENV var—no code changes.

**Next on my roadmap:**

1. **PDF/DOCX support**: Let users drop in native resume formats.
2. **Parallel pipelines**: Speed up large-scale tests by running multiple applications 
concurrently.
3. **Dashboard UI**: Surface real-time progress and allow inline feedback in a web interface.
4. **Tone customization**: Add sentiment analysis to adapt the cover letter’s style (e.g., 
formal vs. casual).

---

## Looking Ahead: The Future of AI Agents

As foundational LLMs grow ever more capable, agents will not only execute tasks but 
**learn and optimize** over time, forming continuous improvement loops. 
The **World Economic Forum** predicts agents will shoulder specialized coding assignments, 
real-time analytics, and more—closing critical skill gaps in fast-moving 
sectors. With advances in model efficiency, multimodal reasoning, and on-device inference, 
the next wave of AI agents promises autonomy and adaptability far beyond today’s demos. 
Building off Crew AI’s flexible, role-based core, I’m excited to see how powerful agents 
will reshape workflows—and what new possibilities emerge as we hit higher LLM performance 
thresholds.



For more details, check out the full working sample on my 
[GitHub repo](https://github.com/gsantopaolo/cv-pilot)

