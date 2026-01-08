---
title: 'Using Agentic AI to Modernize Large Scale Code'
date: '2025-05-10T10:31:30+00:00'
author: gp
layout: post
image: /content/2025/05/reforgeai.jpg
categories: [Agents, LLM, CrewAI]
---



Modernizing large-scale legacy Java applications in banking is a multifaceted challenge, 
combining intricate domain logic with stringent compliance and security requirements. Generative AI—especially in agentic form—promises to accelerate this process by automating both documentation and code transformation. In the sample **Reforge AI** project, we first leveraged an AI-driven “Documentation Crew” to produce top-notch, Mermaid-rich diagrams and a detailed migration plan, then handed off to a “Gen Code Crew” of agents that seamlessly converted Java EE constructs to Spring Boot at scale. While the approach yields elegant code conversions, LLMs still struggle with UI scaffolding and complex `pom.xml` dependency graphs—good news for UI developers and build engineers alike.

---

## The Kitchensink Story and Scaling Ambition

Our proof of concept centered on migrating the classic JBoss “Kitchensink” sample to Spring Boot, but the real goal is **enterprise-scale modernization** of core banking systems.
Banks typically face:

* **Code ownership silos**, where veteran developers hold tribal knowledge and resist change.
* **Sparse or outdated documentation**, forcing new teams to reverse-engineer services.
* **High compliance bar**, mandating exhaustive traceability and security reviews.

Reforge AI’s two-phase, agentic process addresses these at scale by first generating **authoritative documentation**, then using that as the blueprint for safe, incremental code upgrades.

---

## Phase 1: Documentation – Mermaid Diagrams & Beyond

### 1.1 Automated Architecture Capture

An AI “Documentation Crew” agents scan the codebase and existing Javadoc to infer module boundaries, data flows, and integration points. They output:

* **Sequence and component diagrams** in Mermaid syntax
* **Dependency graphs** of third-party libraries
* **Service catalogs** with endpoint signatures

Mermaid’s simple yet powerful syntax lets teams learn diagrams in a day and embed them directly in Markdown docs ([mermaid.js.org][1]).

### 1.2 Iterative Human-in-the-Loop Refinement

Rather than fully trusting a single LLM pass, Reforge AI uses an **improvement loop**:

1. Engineers review auto-generated docs.
2. Feedback is fed back into agents’ prompts.
3. Agents re-render updated diagrams and text.

This hybrid approach mirrors best practices in developer documentation, where visual elements are critical for complex workflows ([mermaidchart.com][2]). By the end of Phase 1, teams have a **battle-tested migration plan** with rich Mermaid visuals and precise upgrade steps.

---

## Phase 2: Agentic Code Generation at Scale

### 2.1 Crew Configuration & Orchestration

The **Gen Code Crew** sets up dedicated agents for:

* **Code conversion** (EJB → Spring components)
* **Dependency updates** (`pom.xml` to Java 17, Spring Boot 3)
* **Test scaffolding** (unit and integration tests)
* **Compliance checks** (security annotations, style guides)

This multi-agent choreography reflects the broader industry shift towards **AI agents as team members**, not mere autocomplete tools ([The Economic Times][3]).

### 2.2 Seamless Module Rewrites

Agents process one module at a time, yielding:

* **Clean Spring Boot services** with proper `@Service` and `@Repository` annotations
* **Updated build scripts** reflecting modern plugin configurations
* **Auto-generated DTOs** and mapping logic

Despite intricate interdependencies, the code “just compiles” in most cases, thanks to the prior documentation scaffolding and project skeletons provided to agents.

---

## Challenges & Opportunities – UI Gaps and POM Complexities

While code conversion runs smoothly, two pain points emerged:

1. **UI Generation Shortcomings**
   LLMs excel at backend refactoring but falter on intricate frontend layouts and CSS frameworks ([Medium][4]). For large projects, UI developers remain indispensable for crafting pixel-perfect interfaces and interactive components.

2. **`pom.xml` & Dependency Graphs**
   General-purpose LLMs often mishandle Maven’s transitive dependencies and plugin versions, leading to broken builds ([Medium][5]). Specialized pipelines or compiler-aware tooling must complement LLMs to lock down stable dependency trees.

These gaps highlight that, even with advanced agentic AI, **human expertise in UI/UX and build engineering** is critical—ensuring a synergistic human-AI partnership.

---

## Lessons Learned & Best Practices

1. **Phased Incrementalism**
   Break migrations into logical slices (e.g., account services, transaction services). This reduces risk and aligns with banking de-risking strategies ([FintechOS][6]).

2. **Mermaid-First Documentation**
   Embedding diagrams directly in Markdown ensures docs stay close to the code and are easy to update. Teams can learn Mermaid in a single day and rapidly visualize new designs ([mermaid.js.org][1]).

3. **Agentic Orchestration**
   Use multiple specialized agents (conversion, testing, compliance) rather than a monolithic prompt. This mirrors best practices in AI-first transformations ([The Official Microsoft Blog][7]).

4. **Human-in-the-Loop Guardrails**
   Regular reviews catch hallucinations and ensure alignment with security and compliance requirements. Always validate generated code with static analysis and test suites.

---

## Conclusion

Reforge AI’s agentic approach—anchored by robust documentation with Mermaid diagrams and multi-agent code generation—provides a **scalable blueprint** for modernizing legacy Java systems in banking. While LLMs still need human partners for UI finesse and dependency management, the combined workflow slashes manual effort, mitigates risk, and paves the way for continuous modernization at enterprise scale.

[1]: https://mermaid.js.org/intro/getting-started.html?utm_source=genmind.ch "Mermaid User Guide"
[2]: https://www.mermaidchart.com/blog/posts/7-best-practices-for-good-documentation/?utm_source=genmind.ch "7 best practices (+ examples) for good developer documentation"
[3]: https://economictimes.indiatimes.com/tech/artificial-intelligence/big-in-big-tech-ai-agents-now-code-alongside-developers/articleshow/121390787.cms?utm_source=genmind.ch "Big in big tech: AI agents now code alongside developers"
[4]: https://medium.com/%40adnanmasood/code-generation-with-llms-practical-challenges-gotchas-and-nuances-7b51d394f588?utm_source=genmind.ch "Code Generation with LLMs: Practical Challenges, Gotchas, and ..."
[5]: https://medium.com/%40jelkhoury880/why-general-purpose-llms-wont-modernize-your-codebase-and-what-will-eaf768481d38?utm_source=genmind.ch "Why General-Purpose LLMs Won't Modernize Your Codebase ..."
[6]: https://fintechos.com/blogpost/how-to-de-risk-core-modernization-in-banking/?utm_source=genmind.ch "5 Strategies for De-risking Core Modernization in Banking - FintechOS"
[7]: https://blogs.microsoft.com/blog/2025/04/28/how-agentic-ai-is-driving-ai-first-business-transformation-for-customers-to-achieve-more/?utm_source=genmind.ch "How agentic AI is driving AI-first business transformation for ..."

---

## Need Help with Your AI Project?

Whether you're building a new AI solution or scaling an existing one, I can help. Book a free consultation to discuss your project.

[Book a Free Consultation](https://calendar.app.google/QuNua7HxdsSasCGu9){: .btn .btn-primary}
