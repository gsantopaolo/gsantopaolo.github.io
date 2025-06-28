---
title: 'How to Supercharge Your Terminal with Gemini CLI'
date: '2025-06-28T6:31:30+00:00'
author: gp
layout: post
image: /content/2025/06/gemini-cli.png
categories: [Gemini-Cli, AI Tools]
math: true
mermaid: true
---


## TL;DR

Gemini CLI is Google’s free, open-source AI agent for your terminal. Powered by Gemini 2.5 Pro 
(with a huge 1 million-token context), it lets you scaffold apps, debug and refactor code, 
fetch and summarize web content, automate file ops, generate docs or slides, and even build 
custom chat agents—all via natural-language prompts. 
Install via `homebrew`, `npx`, `npm`; log in with your Google account for **60 req/min & 1 000 req/day free**, 
or supply an API key for higher quotas and full privacy. 
Customize per-project behavior with **gemini.md** and **.geminiignore**, and extend capabilities through 
external MCP servers like Imagen, DuckDuckGo or HuggingFace.

---

## Installation

### Install the `gemini-cli` command system-wide.
 **Homebrew (macOS/Linux)**:

   ```bash
   brew install gemini-cli
   ```
### Install via npx or npm
**Prerequisite**: Node.js ≥ 18 (check your current node version with `node -v`).

   ```bash
   npx https://github.com/google-gemini/gemini-cli?utm_source=genmind.ch
   ```
   ```bash
   npm install -g @google/gemini-cli
   ```
---

## Authentication & Privacy
You can use your Google account (for free) or with an API key, here is how and key differences:
* **Google-account login** (free tier): On first run, pick a color theme then sign in with your personal Google account—no billing setup required. You get **60 model requests/min** and **1 000 requests/day** at no cost. But here is the caveat: Under the [Gemini Code Assist Privacy Notice](https://developers.google.com/gemini-code-assist/resources/privacy-notice-gemini-code-assist-individuals?utm_source=genmind.ch), prompts & responses may be logged for product improvement (opt-out available).
* **API-key login** (usage-based, full privacy): Generate a key in [Google AI Studio](https://console.cloud.google.com/ai/studio?utm_source=genmind.ch), then run:

  ```bash
  export GEMINI_API_KEY="YOUR_API_KEY"
  ```

  to switch to API-key mode—enabling higher rate limits, specific‐model access, and **no** data ingestion for improvements.

---

## Project Configuration

* **`.geminiignore`**: Exclude files/folders (like logs or large assets) from being read into context—just like `.gitignore`.
* **`gemini.md`**: Drop this in your repo root to provide persistent “memory” (project rules, styles, or context) that loads automatically each session.
* **`settings.json`**: In `.gemini/`, configure external MCP servers, default models, or bug filing behavior; see the [repo’s docs/cli/configuration.md](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md?utm_source=genmind.ch) for details. 

---

## Getting Started

In any directory—new or existing—run:

```bash
gemini
```

to enter the interactive REPL (`gemini>`). From here, type natural-language prompts like:

```text
> Scaffold a FastAPI todo app with SQLite
```
> If you are planning to use the tool as a code assistant better if you set your default working folder with the @ command eg '@/Users/gp/Developer/samples/gemini-cli-test'
{: .prompt-tip }

and watch it create files, install deps, run tests, and show you the results.

![gemini-cli in action](/content/2025/06/gemini-cli1.png){: width="500" height="300" }
_gemini-cli in action_

After a while the tool was able to create a fully, almost, working solution.
It just forget to create the `__init__.py` so uviconr was failing to start.
A quick look and after 10 seconds the APis were working!
Then I asked to create a simple python file to test the APIs.
It worked first shot. I asked some more changes to the tested and everything went well.
In less than ten minutes I had the solution working and even well organized and architected.
<br />
This is the proof that Andrej Karpathy is a visionary and he is totally right about where software development is going 
see his post from 2017 [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35?utm_source=genmind.ch), and his 
speech at the AI Startup School in San Francisco [Software Is Changing (Again)](https://www.youtube.com/watch?v=LCEmiRjPEtQ?utm_source=genmind.ch).
I'm no one, but I can't agree more with him, I saw this happening back in 2019 when I gave my first speech about AI [AI for every developer](https://www.youtube.com/watch?v=ujhtW6UGvhM?utm_source=genmind.ch)
<br />
I diverged a bit, but I'm really exited about the coming revolution in software development. Now back to track...


---

## Core Slash Commands & Built-In Tools

Type `/tools` in the REPL to see everything at your fingertips. Below is the  slash-command list:

| Command           | Description                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `/bug`            | File a GitHub issue with your current prompt as the title (configurable via `settings.json`).  |
| `/chat save`      | Save the current conversation under a named tag.                                               |
| `/chat resume`    | Resume a previously saved conversation.                                                        |
| `/clear`          | Clear the terminal display (same as Ctrl+L).                                                   |
| `/compress`       | Replace the entire context with a concise summary to save tokens.                              |
| `/help`           | Show general help and usage tips.                                                              |
| `/mcp`            | List or describe configured Model Context Protocol servers and their tools.                    |
| `/memory add`     | Add a fact to your project memory (persisted via `GEMINI.md`).                                 |
| `/memory show`    | Display all saved memory entries.                                                              |
| `/memory refresh` | Re-load memory from `GEMINI.md`.                                                               |
| `/restore`        | Roll back all file changes to the state before your last tool invocation.                      |
| `/stats`          | Display token usage, cache savings, and session duration (cached tokens only in API-key mode). |
| `/theme`          | Switch the CLI’s color theme.                                                                  |
| `/auth`           | Toggle authentication method (OAuth vs API key).                                               |
| `/about`          | Show version info, authentication status, and environment details.                             |
| `/quit`           | Exit the Gemini CLI.                                                                           |

---

## What You Can Do: Sample Use Cases

### **Coding Assistant**

* **Prompt**:

  ```text
  > Generate a Next.js + Tailwind CDN landing page for a pizza by the slice bakery like the ones in Rome.
  ```

  *Outcome*: Creates `pages/index.js`, links Tailwind from CDN, and scaffolds responsive components.

* **Prompt**:

  ```text
  > Find and fix failing pytest tests in tests/test_user.py.
  ```

  *Outcome*: Applies patch edits, runs `!pytest`, and reports success.

### **Research & Summaries**

* **Prompt**:

  ```text
  > Web-fetch "https://techcrunch.com" and save summaries of the top 5 AI articles to ai_summaries.txt.
  ```

  *Outcome*: Downloads pages, extracts headlines, writes summaries via `write-file`. ([datacamp.com][3])

### **Content Generation**

* **Prompt**:

  ```text
  > Draft a 500-word GenMind.ch blog post on the latest trends in LLM fine-tuning.
  ```

  *Outcome*: Outputs a structured draft ready to copy into your CMS. 

### **Automation & DevOps**

* **Prompt**:

  ```text
  > !for f in *.png; do convert "$f" "${f%.png}.webp"; done
  ```

  *Outcome*: Batch-converts all PNGs to WebP via shell integration.

### **Slide Decks**

* **Prompt**:

  ```text
  > Create a 5-slide PowerPoint deck summarizing last week’s Git commits by author.
  ```

  *Outcome*: Uses a Python script under the hood to parse `git log` and outputs `deck.pptx`. ([datacamp.com][3])

### 6. **Custom Agents (ADK)**

* **Prompt**:

  ```text
  > Using DuckDuckGo MCP, scaffold an ADK agent that answers questions about this codebase.
  ```

  *Outcome*: Generates `agent.py`, Docker setup, and config for `adk web`. ([cloud.google.com][2])

---

## Advanced: Extending with MCP Servers

1. **Configure** `.gemini/settings.json` with your MCP endpoints (e.g., Imagen, DuckDuckGo, Hugging Face).
2. **Install** local clients (`pip install duckduckgo-mcp-server`).
3. **List** with `/mcp list` and **describe** with `/mcp desc`.
4. **Invoke** in prompts, e.g.:

   ```text
   > /mcp huggingface-image generate an illustration of a data scientist
   ```

---

## Tips & Best Practices

* **Use `gemini.md`** to seed long-term context and project rules. ([cloud.google.com][2])
* **Lean on `web-fetch`** and `/search` for real-time facts.
* **Persist insights** with `/memory add`.
* **Avoid token bloat** via `/compress` on stale context.
* **Monitor limits** daily with `/stats` to stay under free quotas.

---

## Wrap-Up

Gemini CLI brings an incredibly versatile AI assistant to your terminal—whether you’re coding, 
researching, automating, or building agents. It’s free to try, simple to install, 
and endlessly extensible. Give it a spin:

   ```bash
   brew install gemini-cli
   ```

For full docs and the source code, check out the [GitHub repo](https://github.com/google-gemini/gemini-cli?utm_source=genmind.ch) and the [official docs](https://cloud.google.com/gemini/docs/codeassist/gemini-cli?utm_source=genmind.ch).

Happy prompting!

[1]: https://github.com/google-gemini/gemini-cli?utm_source=genmind.ch "google-gemini/gemini-cli: An open-source AI agent that ... - GitHub"
[2]: https://cloud.google.com/gemini/docs/codeassist/gemini-cli?utm_source=genmind.ch "Gemini CLI  |  Gemini for Google Cloud"
[3]: https://www.datacamp.com/tutorial/gemini-cli?utm_source=genmind.ch "Gemini CLI: A Guide With Practical Examples - DataCamp"
[4]: https://github.com/reugn/gemini-cli?utm_source=genmind.ch "reugn/gemini-cli: A command-line interface (CLI) for Google ... - GitHub"
[5]: https://github.com/google-gemini/gemini-cli/discussions/2301?utm_source=genmind.ch "Gemini CLI v0.1.6 Introduces New Privacy Command, Improved ..."
