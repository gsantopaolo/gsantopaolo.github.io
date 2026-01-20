---
layout: page
icon: fas fa-code
order: 6
---

Production AI systems built for scale, deployed in real-world environments serving millions of users. Each project demonstrates end-to-end engineering—from research prototypes to production-grade microservices.

---

## CogniX – Enterprise RAG Platform
**Open-source Retrieval-Augmented Generation (RAG) platform** for semantic search over millions of documents. Built as modular, production-ready system with enterprise data connectors, vector databases (Qdrant/Milvus), and secure multi-tenant architecture. Deployed in production for real customers with LangChain/LangGraph orchestration and distributed inference on multi-GPU clusters.

**Tech Stack:** Python, PyTorch, Kubernetes, NATS, CockroachDB, MinIO, Grafana/Prometheus  
**Blog:** [Cognix – An Enterprise Degree RAG System](https://genmind.ch/posts/cognix-an-enterprise-degree-rag-system/)  
**GitHub:** [github.com/gen-mind/cognix](https://github.com/gen-mind/cognix)

---

## ReforgeAI – Agentic Legacy Modernization
**Agentic AI system for modernizing legacy Java codebases** into Spring Boot at scale. Automatically analyzes code, generates documentation with Mermaid diagrams, creates transformation plans, and executes refactoring tasks (dependency updates, framework migrations, security hardening) with human-in-the-loop feedback. Uses CrewAI multi-agent orchestration. Deployed in production for real customer modernizations.

**Tech Stack:** Python, CrewAI, LangChain, GPT-4  
**Blog:** [Using Agentic AI to Modernize Large Scale Code](https://genmind.ch/posts/Using-Agentic-AI-To-Modernize-Large-Scale-Code/)  
**GitHub:** [github.com/gsantopaolo/reforge-ai](https://github.com/gsantopaolo/reforge-ai)

---

## Sentinel-AI – Real-Time Event Monitoring
**Production-grade event-driven platform** that ingests, filters, ranks, and detects anomalies in IT event streams using embeddings and LLMs. Scales to millions of users with microservices on Kubernetes, NATS JetStream for message bus, and Qdrant for vector search. Deployed as continuous monitoring system with Grafana/Prometheus/Loki observability. Features agentic agent orchestration via CrewAI for log analysis and incident triage.

**Tech Stack:** Python, NATS, Qdrant, Postgres, Kubernetes, React, CrewAI  
**Blog:** [Sentinel-AI - Designing a Real-Time, Scalable AI Newsfeed](https://genmind.ch/posts/Sentinel-AI-Designing-a-Real-Time-Scalable-AI-Newsfeed/)  
**GitHub:** [github.com/gsantopaolo/sentinel-AI](https://github.com/gsantopaolo/sentinel-AI)

---

## CreativeCampaign-Agent – AI Creative Automation
**Agentic system for social advertising creatives** that generates brand-safe imagery, intelligently adds logos using GPT-4o Vision, localizes copy across markets, and exports multi-format assets. Event-driven microservices (5 specialized agents) orchestrated via NATS with DALL·E 3 + GPT-4o-mini. Deployed in production to automate creative workflows at scale. Generates 32 campaign variants in under 10 minutes for ~$1.50.

**Tech Stack:** Python, NATS, MongoDB, MinIO/S3, OpenAI APIs, FastAPI, Streamlit  
**Blog:** [Building an AI-Powered Creative Campaign Pipeline](https://genmind.ch/posts/Building-an-AI-Powered-Creative-Campaign-Pipeline/)  
**GitHub:** [github.com/gsantopaolo/CreativeCampaign-Agent](https://github.com/gsantopaolo/CreativeCampaign-Agent)

---

## DeltaE – Automated Color Correction for Fashion AI
**Production-ready color fidelity pipeline** for AI-generated fashion photography. Ensures ΔE2000-accurate garment colors while preserving texture and material appearance. Combines advanced segmentation (Segformer), hybrid color correction (optimal transport + LCh space), and comprehensive quality metrics (SSIM, spatial coherence index). Deployed for fashion e-commerce at scale. Processes images in 1.5s with 80% automated success rate.

**Tech Stack:** Python, PyTorch, Segformer, Computer Vision (OpenCV, Pillow)  
**Blog:** [DeltaE: Automated Color Correction for AI Generated Fashion Photography](https://genmind.ch/posts/DeltaE-Automated-Color-Correction-for-AI-Generated-Fashion-Photography/)  
**GitHub:** [github.com/gsantopaolo/DeltaE](https://github.com/gsantopaolo/DeltaE)

---

## Multilingual, Multimodal Vector Search
**Scalable semantic search system** using Qdrant and advanced embedding models, enabling rich search experiences across languages and content types (text, images, audio). Deployed for user base exceeding **100 million users** with production-grade performance, multilingual support, and cross-modal retrieval capabilities.

**Tech Stack:** Python, Qdrant, Hugging Face Transformers, Kubernetes  
**Related:** [RAG vs. CAG: A New Frontier in Augmenting Language Models](https://genmind.ch/posts/RAG-vs-CAG-A-New-Frontier-in-Augmenting-Language-Models/)

---

## Additional Projects

### AI Code Generator for Developer Productivity
Internal tool that analyzes existing codebases to generate repetitive code blocks and boilerplate, integrated into development workflows to improve velocity. Uses LLM-based code understanding and generation.

### AI Agents & Bots (Agentic Systems)
Suite of production AI agents for industries including real estate, travel, and finance. Includes WhatsApp bot for appointment scheduling and AI-driven trading bot with external API orchestration and business rules.

**Related:** [Top Agentic AI Frameworks](https://genmind.ch/posts/Top-Agentic-AI-Frameworks/)

### Fine-tuning & Distributed AI R&D
Research and production work on fine-tuning LLMs (Pixtral multimodal text+vision) and diffusion models (Flux, Stable Diffusion) for business applications. Distributed inference across multi-GPU setups using vLLM and Hugging Face TGI. Experiment tracking with Weights & Biases/TensorBoard and evaluation pipelines using Inspect AI and LangTrace.

---

## Need Help with Your AI Project?

Looking to build a similar production AI system for your organization? Book a free consultation to discuss how I can help.

[Book a Free Consultation](https://calendar.app.google/QuNua7HxdsSasCGu9){: .btn .btn-primary}

---

> All projects demonstrate production engineering patterns: event-driven microservices, horizontal scaling, fault tolerance, observability, and comprehensive testing. Built for real-world deployment serving millions of users.
{: .prompt-info }
