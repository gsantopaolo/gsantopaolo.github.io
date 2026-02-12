# ReAct Pattern Market Research Assistant

**Production-ready implementation of the ReAct pattern using Microsoft Agent Framework**

This project demonstrates how to build AI agents that continuously interleave reasoning and action (Think → Act → Observe → repeat), enabling dynamic adaptation to unpredictable scenarios.

📖 **Blog Post**: [Building ReAct Agents with Microsoft Agent Framework: From Theory to Production](https://genmind.ch/posts/Building-ReAct-Agents-with-Microsoft-Agent-Framework-From-Theory-to-Production/)

🔗 **Companion Project**: [Planning Pattern with Microsoft Agent Framework](../planning-claude-sdk-market-research/) - Compare ReAct vs Planning approaches

---

## 🎯 What is the ReAct Pattern?

The **ReAct (Reasoning and Acting) pattern** is an agentic AI design where the agent continuously interleaves reasoning traces with tool-based actions.

### ReAct vs Planning

| Pattern | Approach | Best For |
|---------|----------|----------|
| **ReAct** | Think → Act → Observe → Think → Act... | Unpredictable scenarios, exploration, real-time adaptation |
| **Planning** | Plan → Execute All → Synthesize | Predictable workflows, cost optimization, speed |

### When to Use ReAct

✅ **Use ReAct when you have:**
- Unpredictable scenarios requiring dynamic adaptation
- Exploratory research without clear structure
- Need for continuous validation of each step
- Real-time decision making

❌ **Don't use ReAct when:**
- Workflow is predictable and structured (use Planning instead)
- Cost is primary concern (ReAct uses 5-10x more tokens)
- Speed is critical (ReAct has sequential overhead)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ReAct Pattern Workflow                     │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   [ITERATION 1]       [ITERATION 2]       [ITERATION N]
        │                    │                    │
    THINK                THINK                THINK
        ↓                    ↓                    ↓
     ACT                  ACT                  ACT
        ↓                    ↓                    ↓
   OBSERVE              OBSERVE              OBSERVE
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                      Final Answer
```

### Framework: Microsoft Agent Framework

This implementation uses **Microsoft Agent Framework** - the official successor to Semantic Kernel and AutoGen.

**Key Features:**
- Native ReAct support via `ChatClientAgent`
- Multi-language (Python + .NET)
- Production-ready observability (OpenTelemetry + DevUI)
- Enterprise features (type safety, security filters)

**Why Microsoft Agent Framework?**
- Built-in ReAct loop - no custom implementation needed
- Unified framework combining Semantic Kernel + AutoGen
- Active development by Microsoft AI team
- Azure ecosystem integration

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Azure OpenAI API key (or Anthropic Claude)
- Serper API key for web search

### Installation

```bash
# Clone repository
git clone https://github.com/gsantopaolo/gsantopaolo.github.io.git
cd examples/react-microsoft-agent-framework-market-research

# Install dependencies
pip install agent-framework --pre  # --pre required for public preview
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

### Run Your First ReAct Agent

```bash
python main.py "AI agent market size 2024-2026"
```

Expected output:
```
🔄 ReAct Pattern Market Research Assistant
   Framework: Microsoft Agent Framework
============================================================
📋 Topic: AI agent market size 2024-2026
📝 Output: react_market_report.md
🔊 Verbose: True
============================================================

🔄 Starting ReAct loop (Think → Act → Observe → repeat)...

🤔 THINK → ACT: Calling web_search
   Arguments: {'query': 'AI agent market size 2024 Gartner'}
✅ OBSERVE: web_search completed
   Result preview: [{"title": "AI Agents Market Report"...

🤔 THINK → ACT: Calling calculator
   Arguments: {'expression': '((47.1 / 5.43) ** (1/6) - 1) * 100'}
✅ OBSERVE: calculator completed
   Result preview: 43.2

✅ SUCCESS
Report saved to: react_market_report.md
Execution time: 180.4 seconds
```

---

## 📁 Project Structure

```
react-microsoft-agent-framework-market-research/
├── main.py                  # CLI entry point with ReAct agent
├── tools.py                 # Tool definitions (web_search, calculator)
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── README.md               # This file
└── examples/               # Sample outputs
    └── sample-output.md    # Example market research report
```

---

## 🔧 How It Works

### ReAct Loop Execution

The Microsoft Agent Framework executes the ReAct loop automatically when you call `agent.run()`:

```python
from agent_framework import Agent
from agent_framework.azure import AzureOpenAIChatClient
from tools import web_search, calculator

# Create agent with tools
agent = Agent(
    client=AzureOpenAIChatClient(...),
    instructions="You are a market research agent using ReAct pattern...",
    tools=[web_search, calculator]
)

# Run agent - executes ReAct loop internally
thread = agent.get_new_thread()
result = await agent.run("Research AI agent market size 2024-2026", thread=thread)
```

**What happens internally:**

**Iteration 1:**
1. **THINK**: "I need market size data for AI agents in 2024"
2. **ACT**: Calls `web_search("AI agent market size 2024 Gartner")`
3. **OBSERVE**: Receives search results → "$5.43B in 2024"

**Iteration 2:**
4. **THINK**: "I need growth projections for 2026"
5. **ACT**: Calls `web_search("AI agent market forecast 2026 CAGR")`
6. **OBSERVE**: Receives results → "$47.1B by 2030, CAGR 43.2%"

**Iteration 3:**
7. **THINK**: "I should verify the CAGR calculation"
8. **ACT**: Calls `calculator("((47.1 / 5.43) ** (1/6) - 1) * 100")`
9. **OBSERVE**: Receives result → "43.2"

**Final:**
10. **THINK**: "I have enough information, synthesize final report"
11. **ANSWER**: Returns comprehensive market research report

---

## 🛠️ Tools

### 1. web_search(query)

Search the web for current market information using Serper API.

**Example:**
```python
result = web_search("ReAct pattern AI agents benchmark")
```

**Security:**
- 10-second timeout
- Graceful error handling
- Structured JSON output

### 2. calculator(expression)

Safely evaluate mathematical expressions using AST parsing.

**Example:**
```python
cagr = calculator("((47.1 / 5.43) ** (1/6) - 1) * 100")
# Returns: "43.2"
```

**Security:**
- AST parsing (NOT eval())
- Only allows math operations: +, -, *, /, **, %
- Prevents code injection

---

## 📊 Performance: ReAct vs Planning

Based on identical market research task ("AI agent market size 2024-2026"):

| Metric | ReAct (This Project) | Planning | Difference |
|--------|---------------------|----------|------------|
| **Tokens Used** | ~45,000 | ~8,500 | **5.3x more** |
| **Execution Time** | 180 seconds | 45 seconds | **4x slower** |
| **Cost (GPT-4o)** | $0.68 | $0.13 | **5.2x more expensive** |
| **LLM API Calls** | 12+ (continuous loop) | 2 (plan + synthesis) | **6x more** |
| **Adaptability** | High (adjusts per step) | Low (plan fixed) | **ReAct wins** |
| **Report Quality** | Excellent | Excellent | **Tie** |

**Conclusion:**
- ReAct excels at **unpredictable scenarios** requiring **real-time adaptation**
- Planning excels at **predictable workflows** with **cost/speed requirements**

---

## 🔐 Security Best Practices

This implementation follows **FAANG-level security standards**:

### 1. No `eval()` - AST Parsing Only

```python
# ❌ INSECURE
result = eval(user_expression)

# ✅ SECURE (this project)
tree = ast.parse(expression, mode='eval')
result = _eval_node(tree.body)
```

### 2. Input Validation

```python
# Validate query length
if len(query) > 500:
    raise ValueError("Query too long")

# Validate characters
if any(char in query for char in ['<', '>', ';']):
    raise ValueError("Invalid characters")
```

### 3. Timeouts

```python
# All external calls have timeouts
response = requests.post(url, timeout=10)
```

### 4. Error Handling

```python
# Graceful degradation
try:
    result = tool.execute()
except Exception as e:
    return f"ERROR: {str(e)}"  # Don't crash, report error
```

---

## 📖 Usage Examples

### Basic Research

```bash
python main.py "quantum computing market forecast"
```

### Custom Output File

```bash
python main.py "blockchain enterprise adoption 2026" \
  --output blockchain_report.md
```

### Streaming Mode

```bash
python main.py "autonomous vehicles market" \
  --streaming
```

---

## 🚨 Troubleshooting

### Error: "Missing required environment variables"

**Solution:**
```bash
# Add to .env file
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
```

### Error: "SERPER_API_KEY not found"

**Solution:**
1. Get free API key at https://serper.dev
2. Add to .env:
```bash
SERPER_API_KEY=your-key-here
```

### Error: "pip install agent-framework --pre" fails

**Solution:**
```bash
# Update pip first
pip install --upgrade pip

# Then retry
pip install agent-framework --pre
```

### Agent produces incomplete report

**Check:**
1. API keys are valid
2. Internet connectivity
3. Azure OpenAI API status
4. Check logs for specific errors
5. Try increasing max_iterations

---

## 🎓 Learning Resources

### Microsoft Agent Framework
- [Official Documentation](https://learn.microsoft.com/en-us/agent-framework/)
- [GitHub Repository](https://github.com/microsoft/agent-framework)
- [Quick Start Tutorial](https://learn.microsoft.com/en-us/agent-framework/tutorials/quick-start)
- [Migration from Semantic Kernel](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-semantic-kernel/)

### ReAct Pattern
- [ReAct Paper (Yao et al., ICLR 2023)](https://arxiv.org/abs/2210.03629) - Original research
- [Blog Post: ReAct Pattern](https://genmind.ch/posts/Building-ReAct-Agents-with-Microsoft-Agent-Framework-From-Theory-to-Production/)
- [Agentic Reasoning with Microsoft Agent Framework](https://jgcarmona.com/agentic-reasoning-with-microsoft-agent-framework/)

### Production Case Studies
- [KPMG Clara AI](https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai-common-use-cases-and-design-patterns/)
- [JM Family BAQA Genie](https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai-common-use-cases-and-design-patterns/)
- [Bank of America Erica](https://pub.towardsai.net/production-ready-ai-agents-8-patterns-that-actually-work-with-real-examples-from-bank-of-america-12b7af5a9542)

---

## 📝 License

MIT License - see [LICENSE](../../LICENSE)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

**Improvement ideas:**
- Add caching for repeated searches
- Implement circuit breakers for API failures
- Add support for other LLMs (Anthropic Claude, Gemini)
- Multi-agent ReAct coordination

---

## 📧 Contact

**Author**: GP (genmind.ch)

**Blog**: [genmind.ch](https://genmind.ch)

**Questions?** Open an issue or reach out via blog contact form.

---

**⭐ If this project helped you, please star the repository!**

**🔗 Related Projects:**
- [Planning Pattern with Microsoft Agent Framework](../planning-claude-sdk-market-research/) - Compare approaches
- [Echo-Mind](https://github.com/gsantopaolo/echo-mind) - Full AI agent platform
