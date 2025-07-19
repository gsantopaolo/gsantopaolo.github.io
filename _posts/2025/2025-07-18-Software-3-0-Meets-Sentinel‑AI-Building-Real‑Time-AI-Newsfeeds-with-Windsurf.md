---
title: "Software 3.0 Meets Sentinel‑AI: Building Real‑Time AI Newsfeeds with Windsurf"
date: "2025-07-18T06:31:30+00:00"
author: "gp"
layout: "post"
image: "/content/2025/07/software30.png"
categories: [Windsurf, AI Tools]
published: true
---



In this post, I'll try to show how **Windsurf’s Cascade AI** transforms prompt-driven development—what 
Andrej Karpathy calls **Software 3.0**, where “prompts are programs”—into real, end‑to‑end scaffolding, 
refactoring, and maintenance of [Sentinel‑AI](https://genmind.ch/posts/Sentinel-AI-Designing-a-Real-Time-Scalable-AI-Newsfeed/). 
<br/>
You’ll see how to set up inline AI rules for Python, scaffold FastAPI services, perform multi‑file edits 
(e.g., adding logging), generate complex Qdrant initialization code, apply surgical in‑line 
tweaks (with extra care around shared `lib_py` utilities), 
automate CLI deployments, and one‑click refactors and docs via codelenses—all within 
Windsurf’s agentic IDE. 


---

## Introduction

Windsurf’s Cascade AI embodies **Software 3.0**, a paradigm (software 3.0) popularized by [Andrej Karpathy](https://www.youtube.com/watch?v=LCEmiRjPEtQ) 
where **natural‑language prompts** become the primary interface for writing and 
refactoring code. 
<br />
Karpathy argues we’ve moved from **Software 1.0** (hand‑written code) 
through **Software 2.0** (machine‑learned parameters) to **Software 3.0**, in which developers 
“just ask” and LLMs rewrite legacy codebases over time. 
Windsurf’s Cascade lets you **vibe code**—iteratively prompt, verify, 
and refine—so you focus on guiding the AI rather than typing every line 
yourself.
<br />
Sentinel‑AI is a production‑grade Python microservices AI software for real‑time news ingestion, 
filtering, ranking, and anomaly detection, built on **NATS JetStream**, **Qdrant**, and 
**FastAPI**. 
In the sections below, I'll map Windsurf’s seven game‑changing features to hands‑on 
actions in Sentinel‑AI’s codebase.

---

## Configuring Windsurf with Software 3.0 Principles

First, codify your stack once via **Cascade Memories & Rules** so 
Windsurf “remembers” you’re working with Python 3.12, FastAPI, NATS, and 
Qdrant. In Cascade’s Settings → Rules panel, create a **workspace rule**:

```yaml
language: Python 3.10+
framework: FastAPI
message_bus: NATS JetStream
vector_db: Qdrant
important: if you make changes in any of lib_py you need to make sure this change does not affect other code using it lib py it is a shared code
```

This upfront rule‑definition aligns with Karpathy’s “prompts as programs” by 
embedding your tech stack into the AI’s context, saving tokens and latency on 
every prompt.

![Windsurf Rules pane](/content/2025/07/rulepanel.png){: width="500" height="300" }
_Windsurf Rules pane_

---

## Feature 1: Cascade Memories → Environment Setup

First of all, if you are using Windsurf for the first time on a project that already contains code and docs, 
I use a prompt to drive Windsurf to analyze and understand the whole project, something like:
```bash
Read all the docs inside the folder and its subfolder (search recursively)
Read excluded/spec, and all the md in docs, read readme.md in the root folder that you have access.
If you can't read even one of the docs I asked to, you stop and you tell me, do not go any further 
if you cannot read the files I mentioned.
At this point you shall have a pretty good understanding of what Sentinel AI is.
Important: there can be some missmatch between the documentation you read and the code. 
Code win, documentation might be a bit outdated.
Ok now read all the code in src, all the services, and also lib_py which contains shared 
classes and functions used in all the services.
When you are done I want evidences that you understood the project. 
Make a summary of the documentation, and a quick summary of every service in src as well as for lib_py
```
![Windsurf analyzing the hole repo](/content/2025/07/analyze1.png){: width="500" height="300" }
_Windsurf analyzing the hole repo_


After that we shall start working.
<br/>
Use **Write mode** to generate consistent `.env.example` files across services (windsurf isn't able to
read and write files and folders listed in the .gitignore file):

* **Prompt:** “Create `deployment/api.env.example` with `NATS_URL`, `STREAM_RAW_EVENTS=raw.events`, and `OPENAI_API_KEY`.”
* **Outcome:** Cascade writes a complete `api.env.env` matching Sentinel‑AI’s `deployment/.env.example`, 
* including `STREAM_FILTERED_EVENTS` and other required keys.

And here you go, you click "Apply" and then "Accept" and your file it is automatically created!r 


![Environment Setup](/content/2025/07/apienv.png){: width="500" height="300" }
_Environment Setup_
---

## Feature 2: Write vs. Chat – Scaffolding & Exploration

Leverage **Write** and **Chat** modes interchangeably for code generation and understanding:

* **Write mode:** “Scaffold `src/api/main.py` as a FastAPI app with `/events/raw`, `/events/filtered`, and `/events/ranked` endpoints.” Cascade generates imports, Pydantic models, and NATS subscriber logic in under a minute.
* **Chat mode:** “Explain how `connector/main.py` polls RSS feeds and publishes to `raw.events`.” Windsurf analyzes the file and returns a concise architecture overview.

This generate‑verify loop is central to Karpathy’s **vibe coding** method—prompt, inspect, 
and re‑prompt until satisfied.

*Screenshot suggestion:* Split‑pane: left shows Write prompt, right shows generated `main.py`.
![Windsurf explaining connector](/content/2025/07/analyze2.png){: width="500" height="300" }
_Windsurf explaining connector_
---

## Feature 3: Multi‑File Editing – Bulk Upgrades

For cross‑cutting concerns like logging, Cascade can update multiple files in one session:

* **Prompt:** “Add `logger = get_logger(__name__)` import and initialization at the top of every `src/*/main.py`.”
* **Workflow:** Cascade stages changes across **scheduler**, **connector**, **filter**, **ranker**, **inspector**, **guardian**, and **web** services. You review diffs service by service, approving each before execution.

This showcases AI’s ability to “eat” entire codebases with a single prompt, as Karpathy predicts for Software 3.0 migrations.
---

## Feature 4: Supercomplete – Complex Logic Generation

Supercomplete uses context from all open files to generate multi‑line code:

* **Task:** In `src/filter/filter.py`, after the `async def initialize_qdrant()` stub, type “continue.”
* **Result:** Cascade writes Qdrant client setup, collection creation, batch insertion logic, and ties into `lib_py/utils.py` for shared helpers ([DataCamp][4]).

> **Library Caveat:** Because `lib_py` is a **shared code** bundle used across services, always add to your prompt:
> “Ensure changes in `lib_py/utils.py` do not break other services using it.”
> This prevents unintended global side‑effects.

---

## Feature 5: In‑Line Commands – Surgical Tweaks

Apply precise, localized edits without disturbing surrounding code:

* **Example:** Select `compute_decay()` in `src/ranker/decay.py` and run “Change half‑life parameter from 12h to 6h.”
* **Outcome:** Cascade updates only that function signature and adjusts related imports/tests automatically.

This surgical approach exemplifies how prompts replace manual refactoring in Software 3.0 workflows.

---

## Feature 6: Command in Terminal – Safe Deployments

Have Windsurf propose and verify CLI commands in natural language:

* **Prompt:** “Generate a Helm command to deploy `guardian` to namespace `sentinel-ai`.”
* **Suggestion:**

  ```
  helm upgrade --install guardian ./helm/guardian \
    --namespace sentinel-ai \
    --set image.tag=latest
  ```
* **Verification:** You inspect and approve before running.

Automating deployment commands via prompts mirrors the shift from traditional scripts to conversational ops in Software 3.0, yet with human‑in‑the‑loop safeguards.

---

## Feature 7: Codelenses – Docs & Refactoring

One‑click codelenses let you generate docs and refactor code seamlessly:

* **Add Docstring:** Hover above `async def inspect_event()`, click “Add Docstring,” and Cascade generates a full docstring consistent with your style guide.
* **Refactor:** Use “Refactor” on duplicate validation in `inspector/rules.py` to extract a helper into `lib_py/config.py`.

These instant transformations illustrate Karpathy’s promise that English prompts can orchestrate sophisticated code rewrites.


---

## Bonus: Auto‑execute Settings – Streamlined Workflow

Configure Windsurf’s allow/deny lists so routine commands run without prompts:

* **Allow list:** `docker-compose up`, `pytest --maxfail=1`
* **Deny list:** `rm -rf /`, `kubectl delete namespace sentinel-ai`
* **Model Judgment:** For unrecognized commands, Windsurf prompts—balancing speed with safety, per Karpathy’s “keep AI on a leash” advice .

*Screenshot suggestion:* Windsurf terminal settings panel showing allow/deny lists.

---

## Conclusion & Next Steps

By applying **Karpathy’s Software 3.0** principles with **Windsurf’s Cascade AI**, you can scaffold, refactor, and operate a complex Python microservices pipeline like Sentinel‑AI in minutes rather than days . Try these patterns today:

* Fork the [Sentinel‑AI repo](https://github.com/gsantopaolo/sentinel-AI).
* Share your **vibe‑coding** prompts and recipes.
* Explore autonomous agents in `src/agentic` and push the boundaries of **prompt‑centric programming**.

Happy coding and welcome to **Software 3.0**!

[1]: https://medium.com/data-science-in-your-pocket/software-is-changing-again-96b05c4af061?utm_source=genmind.ch "Andrej Karpathy's Software is Changing (Again) summarized - Medium"
[2]: https://www.nextbigfuture.com/2025/06/software-3-0-by-karpathy.html?utm_source=genmind.ch "Software 3.0 By Karpathy | NextBigFuture.com"
[3]: https://docs.windsurf.com/windsurf/cascade/memories?utm_source=genmind.ch "Cascade Memories - Windsurf Docs"
[4]: https://www.datacamp.com/tutorial/windsurf-ai-agentic-code-editor?utm_source=genmind.ch "Windsurf AI Agentic Code Editor: Features, Setup, and Use Cases"
[5]: https://en.wikipedia.org/wiki/Vibe_coding?utm_source=genmind.ch "Vibe coding"
[6]: https://www.businessinsider.com/openai-cofounder-andrej-karpathy-keep-ai-on-the-leash-2025-6?utm_source=genmind.ch "'Keep AI on the leash' because it's far from perfect, says OpenAI's cofounder Andrej Karpathy"
[7]: https://www.reddit.com/r/theprimeagen/comments/1lf75vt/andrej_karpathy_software_is_changing_again/?utm_source=genmind.ch "Andrej Karpathy: Software Is Changing (Again) : r/theprimeagen"
[8]: https://www.prismetric.com/windsurf-vs-cursor/?utm_source=genmind.ch "Windsurf vs Cursor: Which AI Code Editor is Better? - Prismetric"
[9]: https://www.analyticsvidhya.com/blog/2025/06/andrej-karpathy-on-the-rise-of-software-3-0/?utm_source=genmind.ch "Andrej Karpathy on the Rise of Software 3.0 - Analytics Vidhya"
[10]: https://www.reddit.com/r/Codeium/comments/1gwz0sd/my_experience_with_windsurf_editor_with_cascade/?utm_source=genmind.ch "My Experience with Windsurf Editor with Cascade (Claude Sonnet 3.5)"
[11]: https://www.usesaaskit.com/blog/how-to-set-up-windsurf-ai-code-editor?utm_source=genmind.ch "How to Set Up Windsurf AI Code Editor: A Step-by-Step Guide"
[12]: https://damiandabrowski.medium.com/day-77-of-100-days-agentic-engineer-challenge-windsurf-cascade-813878ab2d32?utm_source=genmind.ch "Day 77 of 100 Days Agentic Engineer Challenge: Windsurf Cascade"
