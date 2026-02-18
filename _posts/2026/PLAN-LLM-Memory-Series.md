---
title: "none"
date: "2026-02-18T12:00:00+01:00"
author: "gp"
layout: "post"
image: "/content/2026/02/llm-memory-part1.png"
categories: [LLM, Memory, AI Agents, Microsoft Agent Framework, Cost Optimization]
published: false
mermaid: true
---

# Phase A — Plan: "LLM Memory Management in Production" (2-Post Series)

**Status:** Awaiting approval before writing
**Date:** 2026-02-18
**Citation style:** [n] footnotes

---

## Post 1: "Your LLM Has Amnesia: A Production Guide to Memory That Actually Works"

**Hook:** "Your chatbot forgets who it's talking to after 15 messages. Your RAG pipeline hallucinates because the relevant answer is buried in token 47,000. You're paying $3.20 per conversation because you send the full history every turn. The context window is not a memory system — it's a scratchpad."

**Target:** ~4,000–4,500 words

### Outline

**§1. Bottom Line First**
- Summary table: strategy × cost × latency × accuracy tradeoff
- Quick decision matrix so readers can skip to what matters

**§2. The Context Window Is Not Memory**
- What happens at 128K tokens (GPT-4o) and 200K (Claude): cost per turn ($0.32 at full 128K), latency scaling
- "Lost in the Middle" problem [1] — >30% accuracy drop for mid-context info (TACL 2024, peer-reviewed)
- Context rot: silent performance degradation before hitting token limit
- **Mermaid diagram:** context window filling up turn-by-turn, showing cost + degradation curve
- **Confidence: High** — peer-reviewed paper + pricing from official docs

**§3. Memory Strategy Taxonomy**
- **Buffer Memory** (full history) — simple, expensive, breaks at ~10 turns
- **Sliding Window** (last k turns) — controlled cost, amnesia for older context
- **ConversationSummaryMemory** — LLM summarizes older turns; sub-linear growth; "contextual drift" risk
- **SummaryBufferMemory** (hybrid) — summary of old + raw recent; best general-purpose tradeoff
- **Entity Memory / Knowledge Graph** — extract structured facts; Zep/Graphiti for temporal KG
- **Mermaid diagram:** decision flowchart — "How many turns?" → strategy selection
- **Table:** strategy comparison (tokens/turn, cost, fidelity, best-for)
- **Confidence: High** — LangChain docs + Zep arXiv paper [2]

**§4. Cost Optimization: When to Summarize vs Retain**
- Decision framework: <10 turns → buffer; 10–50 → summary-buffer; 50+ → external memory
- Mem0 benchmarks: 26K → 1.8K tokens (90% reduction), 91% latency reduction [3]
- LightMem: 117× token reduction, 159× fewer API calls [4]
- GPT-4o vs GPT-4o-mini cost gap (33× cheaper on input) — use mini for summarization calls
- **Table:** cost calculator for 500K requests/month at different strategies
- **Confidence: High** for Mem0 (arXiv paper); **Medium** for LightMem (arXiv preprint)

**§5. Real-World Memory Architectures**
- **ChatGPT Memory**: 4-layer system (session metadata → long-term facts → conversation summaries → current window) — NO vector RAG [5]
- **Google Gemini**: structured user_context with half-lives per category; only on "thinking" models
- **Claude Memory**: file-based markdown (CLAUDE.md), user-editable, project-scoped
- **Mem0** (open source): 26% accuracy improvement over OpenAI baseline, used by Netflix, Lemonade; 186M API calls/quarter [6]
- **Sunflower case study**: 80K users, 70–80% token cost reduction with Mem0 [7]
- **Mermaid diagram:** ChatGPT vs Gemini vs Claude memory architecture comparison — side-by-side flow
- **Confidence: Medium-High** — reverse-engineered for ChatGPT; official for others

**§6. The Microsoft Agent Framework Approach**
- Three memory layers: `InMemoryHistoryProvider` (short-term), `BaseContextProvider` (long-term extraction), Azure Foundry managed memory (cloud)
- Code example: building a `UserInfoMemory` context provider
- Checkpoint system for workflow state persistence
- **Mermaid diagram:** Microsoft Agent Framework 3-tier memory architecture
- **Confidence: High** — Microsoft Learn primary docs [8]

**§7. Key Takeaways** (bulleted, bold-prefixed, 6–8 items)

**§8. CTA + Resources** (cross-link to Post 2)

### Mermaid Diagrams for Post 1
1. Context window cost escalation — token growth per turn
2. Memory strategy decision tree
3. ChatGPT vs Gemini vs Claude memory architecture comparison
4. Microsoft Agent Framework memory layers — 3-tier diagram

---

## Post 2: "RAG, Vector Stores, and the GPU Math Behind LLM Memory"

**Hook:** "You built a RAG pipeline. It retrieves 20 chunks, sends 20,000 tokens to the LLM, and 16 of those chunks are noise. Your RTX 6000 Ada has 48 GB of VRAM — and your KV cache just ate 40 of them. Memory management isn't just a software problem. It's a hardware budget."

**Target:** ~4,000–4,500 words

### Outline

**§1. Bottom Line First**
- Summary: vector store choice × embedding model × GPU fit

**§2. RAG as External Memory**
- RAG = long-term recall that doesn't consume context tokens until needed
- Chunking strategies: recursive (80% of cases), semantic (10× cost for ~3% gain), structure-aware (biggest win for Markdown/HTML)
- Optimal sizes: 256–512 tokens general, 1024 with 15% overlap for factual QA (NVIDIA benchmark [9])
- Advanced patterns: HyDE, Self-RAG, CRAG, Agentic RAG
- **Mermaid diagram:** RAG pipeline — ingest → chunk → embed → store → retrieve → rerank → generate
- **Confidence: High** — NVIDIA blog + multiple concordant sources

**§3. Embedding Models: What to Pick**
- **Table:** OpenAI text-embedding-3 vs Cohere embed-v4 vs Qwen3-Embedding-8B vs BGE-M3
- MTEB leaders: Gemini (~71), Qwen3 (70.58 open-source), Cohere v4 (65.2), OpenAI large (64.6)
- Cost: $0.02/MTok (OpenAI small) vs $0.13 (OpenAI large) vs $0.12 (Cohere) vs free (open-source)
- VRAM for local embeddings: BGE-M3 = 1 GB FP16 (negligible next to LLM weights)
- **Confidence: High** — MTEB leaderboard + official pricing

**§4. Vector Store Showdown**
- **Table:** Chroma vs pgvector vs Qdrant vs Pinecone vs Milvus vs Weaviate vs FAISS
- Performance: QPS, latency at scale
- Cost: Pinecone ($70–500+/month, customer churn) vs pgvector (40–60% lower TCO <500M vectors) vs Qdrant (self-hosted, free tier)
- Decision guide: prototype → Chroma; already Postgres → pgvector; flexibility → Qdrant; billion-scale → Milvus
- **Mermaid diagram:** vector store decision flowchart
- **Confidence: High** — official benchmarks + multiple concordant analyses

**§5. Hybrid Search: The Production Standard**
- Dense + BM25 + Reciprocal Rank Fusion = 26–31% NDCG improvement [10]
- Re-ranking: ColBERT (fast, production) vs cross-encoder (accurate, slow) vs Cohere Rerank (API, +10–25% precision)
- **Table:** re-ranker comparison (latency, accuracy gain, cost)
- **Confidence: High** — multiple benchmark sources agree

**§6. The GPU Math: VRAM, KV Cache, and What Fits**
- KV cache formula: `2 × B × S × L × H_kv × D_head × (bits/8)`
- **Table:** Llama 3.1 8B and 70B KV cache at 4K/32K/128K context
- INT8/FP8 KV quantization: 2× savings, negligible quality loss
- **RTX 4000 Ada SFF (20 GB)**: 8B Q4 + BGE-M3 + 32K context = ~11 GB (fits); 128K = doesn't fit
- **RTX 6000 Ada (48 GB)**: 70B Q4 + BGE-M3 + 8K context = ~37 GB (comfortable); 32K = tight
- GPU-accelerated indexing: Qdrant GPU = 10× faster; FAISS CAGRA = 12.3× faster build
- **Mermaid diagram:** "What fits on your GPU?" — decision tree by VRAM tier
- **Confidence: High** — model cards + measured benchmarks

**§7. Putting It Together: Production Architecture**
- Full pipeline: Microsoft Agent Framework + OpenAI embeddings + Qdrant + hybrid search + ColBERT rerank
- Code example: end-to-end RAG with Microsoft Agent Framework `TextSearchProvider`
- **Mermaid diagram:** full production architecture — end-to-end system diagram
- **Confidence: Medium-High** — synthesized from official docs

**§8. Key Takeaways** (bulleted, bold-prefixed, 6–8 items)

**§9. CTA + Resources** (cross-link to Post 1)

### Mermaid Diagrams for Post 2
1. RAG pipeline flow — ingest → chunk → embed → store → retrieve → rerank → generate
2. Vector store decision flowchart
3. "What fits on your GPU?" — VRAM tier decision tree
4. Full production architecture — end-to-end system diagram

---

## Key Claims + Evidence Summary

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| 1 | >30% accuracy drop for mid-context info | Liu et al. TACL 2024 [1] | High |
| 2 | Mem0: 90% token reduction, 91% latency reduction | arXiv 2504.19413 [3] | High |
| 3 | LightMem: 117× token reduction | arXiv 2510.18866 [4] | Medium |
| 4 | Hybrid search: 26–31% NDCG improvement | Superlinked + Infiniflow [10] | High |
| 5 | Qdrant GPU: 10× faster indexing | Qdrant 1.13 release | High |
| 6 | RTX 4000 Ada: 8B Q4 + 32K fits in 20GB | Calculated from model cards | High |
| 7 | RTX 6000 Ada: 70B Q4 + 8K fits in 48GB | Calculated from model cards | High |
| 8 | ChatGPT uses 4-layer memory, no vector RAG | Reverse-engineered analysis [5] | Medium |
| 9 | MS Agent Framework merges AutoGen + SK | Microsoft Learn official [8] | High |
| 10 | Intercom Fin: 41% avg resolution rate | Intercom official | High |

---

## Candidate Sources (with dates & credibility)

| # | Source | Date | Type |
|---|---|---|---|
| [1] | Liu et al. "Lost in the Middle" — TACL 2024 (arXiv:2307.03172) | Jul 2023 / pub 2024 | Peer-reviewed |
| [2] | Zep/Graphiti — arXiv 2501.13956 | Jan 2025 | Preprint |
| [3] | Mem0 — arXiv 2504.19413 | Apr 2025 | Preprint |
| [4] | LightMem — arXiv 2510.18866 | Oct 2025 | Preprint |
| [5] | ChatGPT Memory reverse engineering — llmrefs.com | 2024 | Secondary |
| [6] | Mem0 TechCrunch funding — $24M | Oct 2025 | High credibility |
| [7] | Sunflower case study — mem0.ai/blog | 2025 | Official case study |
| [8] | Microsoft Agent Framework — learn.microsoft.com | Feb 2026 | Primary (official) |
| [9] | NVIDIA chunking benchmark — developer.nvidia.com | 2024 | Primary (official) |
| [10] | Superlinked hybrid search benchmarks | 2025 | Secondary analysis |
| [11] | Anthropic pricing — platform.claude.com | Feb 2026 | Primary (official) |
| [12] | OpenAI pricing — platform.openai.com | Feb 2026 | Primary (official) |
| [13] | MemGPT paper — arXiv 2310.08560 | Oct 2023 | Peer-reviewed |
| [14] | A-MEM — arXiv 2502.12110 | Feb 2025 | Preprint |
| [15] | Memory in the Age of AI Agents survey — arXiv 2512.13564 | Dec 2025 | Preprint |

---

## Deliverables Summary

| Item | Post 1 | Post 2 | Total |
|---|---|---|---|
| Words | ~4,000–4,500 | ~4,000–4,500 | ~8,000–9,000 |
| Mermaid diagrams | 4 | 4 | 8 |
| Tables | 5–6 | 6–7 | ~12 |
| Code examples | 1–2 (MS Agent Framework memory) | 1–2 (RAG + search) | ~3–4 |
| Citations | ~8–10 | ~8–10 | ~15 unique |
