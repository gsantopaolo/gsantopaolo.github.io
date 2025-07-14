---
title: 'Sentinel-AI - Designing a Real-Time, Scalable AI Newsfeed'
date: '2025-07-13T6:31:30+00:00'
author: gp
layout: post
image: /content/2025/07/sentinel2.png
categories: [AI Clusters, Agentic AI, RAG]
published: true
---


> *“What would a production-grade AI cluster look like if built from scratch for **scale, resilience, and lightning-fast insights**?”*  
> **Sentinel-AI** is my answer.  
> Repo 👉 [github.com/gsantopaolo/sentinel-AI](https://github.com/gsantopaolo/sentinel-AI)

This project implements a **real-time newsfeed platform** that can **aggregate, filter, store, and rank events from *any* source you choose**—RSS, REST APIs, webhooks, internal logs, you name it. The main goal is to showcase how an AI-centric micro-service cluster should be architected to survive production traffic at **millions-of-users scale**.

---

## Core Concepts

🔗 **Dynamic Ingestion**  
Subscribe to any data feed (RSS, APIs, webhooks, etc.) and ingest events the moment they happen.

🧹 **Smart Filtering**  
Mix rule-based heuristics with pluggable ML/LLM models to keep only the content your audience cares about.

⚖️ **Deterministic Ranking**  
Weight “importance × recency” with a configurable algorithm and instantly re-rank via open APIs.

---

## Bird’s-Eye Overview


| Service    | What it Does | Publishes | Persists To |
|------------|--------------|-----------|-------------|
| **Scheduler** | Fires *poll* messages on schedule | `poll.source` | — |
| **Connector** | Fetches each source & normalises events | `raw.events` | — |
| **Filter** | Relevance filter + embeddings | `filtered.events` | Qdrant |
| **Ranker** | Scores events (importance×time) | `ranked.events` | Qdrant |
| **Inspector** | Flags anomalies / fake news | — | Qdrant |
| **API** | Ingest, retrieve, re-rank, CRUD sources | `raw.events`, `new/removed.source` | Postgres, Qdrant |
| **Web** | React dashboard ↔️ API | — | — |
| **Guardian** | Monitors NATS DLQ & alerts | notifications | — |

---

## Why It Scales to Millions

* **Micro-services**: Each responsibility is isolated, stateless, and horizontally scalable.  
* **NATS JetStream**: Ultra-low-latency pub/sub with back-pressure and dead-letter queues baked in.  
* **Vector DB (Qdrant)**: Fast semantic search and payload updates; sharded for linear throughput growth.  
* **Kubernetes-ready**: Health probes, autoscaling, and rolling upgrades out of the box.  
* **Async Python**: Every network-bound task uses `asyncio`, squeezing maximum concurrency per pod.  
* **Deterministic failover**: DLQ + Guardian means no silent data loss—ever.  
* **Docker-Compose** for quick local spins, Helm chart (road-map) for prod clusters.

---

## From “Smart” to **Agentic** 🤖

Today, Sentinel-AI already leverages LLMs for embeddings and optional relevance checks.  
Tomorrow, each micro-service can be converted into an *agent* with its own:

1. **Goal** (e.g., “maintain perfect source coverage”),  
2. **Tools** (HTTP client, vector search, scoring algorithms),  
3. **Memory** (Qdrant / Postgres),  
4. **Self-evaluation loop**.

A starter blueprint lives under [`src/agentic`](https://github.com/gsantopaolo/sentinel-AI/tree/main/src/agentic).  
Swapping the current functions for agentic planners is mostly a wiring exercise—no major rewrite required.

---

## Try It in Two Commands

```bash
git clone https://github.com/gsantopaolo/sentinel-AI.git

cd sentinel-AI 

sudo deployment/start.sh   # brings up the whole stack
```
![Sentinel-AI after deployment/start.sh ](/content/2025/07/sentinel1.png){: width="500" height="300" }
_Sentinel-AI after deployment/start.sh_


Open http://localhost:9000 (Portainer) to watch containers boot
<br/>
![Sentinel-AI logs](/content/2025/07/sentinel3.png){: width="500" height="300" }
_Sentinel-AI logs_

<br/>
Then hit the Web UI to add your first RSS feed.
![Sentinel-AI UI](/content/2025/07/sentinel2.png){: width="500" height="300" }
_Sentinel-AI UI


<br/>
And don't forget to watch services' log on Portainer to see the magic odf a distributed system 
send messages to each other and execute their tasks.


<br/>

> Problems or feature ideas? 👉✉️ Open an issue or reach out on [genmind.ch]&#40;https://genmind.ch/&#41;!)

Closing Thoughts
Sentinel-AI is more than a demo; it’s a template for production-grade, AI-powered event pipelines.
If you need real-time insights, bullet-proof reliability, and the freedom to plug in future agentic intelligence, feel free to fork, extend, and deploy.

Happy hacking! 💡🚀
