---
id: 378
title: 'Milvus: The NATS Compatibility Challenge and Our Move to Qdrant'
date: '2024-04-26T16:53:49+00:00'
author: gp
layout: post
guid: 'https://genmind.ch/?p=378'
permalink: /milvus-the-nats-compatibility-challenge-and-our-move-to-qdrant/
site-sidebar-layout:
    - default
ast-site-content-layout:
    - default
site-content-style:
    - default
site-sidebar-style:
    - default
theme-transparent-header-meta:
    - default
astra-migrate-meta-layouts:
    - set
image: /content/2025/03/milvus.png
categories:
    - RAG
tags:
    - 'vector database'
---

As organizations increasingly demand on‑premises solutions for data security and control, Milvus—a leading open‑source vector database—offers attractive features for managing high‑dimensional data. However, while Milvus provides flexibility with multiple message storage options, its current support for NATS is limited to standalone mode (see [Configure NATS](https://chatgpt.com/c/%EE%88%80cite%EE%88%8232%EE%88%81) and [Before you begin](https://chatgpt.com/c/%EE%88%80cite%EE%88%8230%EE%88%81)). This limitation poses significant challenges for horizontal scalability, ultimately impacting deployment cost and complexity.

## Understanding Milvus and Its Message Storage Options

Milvus leverages message storage systems to manage the logs of recent changes, stream outputs, and subscription data. The Milvus Operator allows users to configure several back‑end message queues:

- **RocksMQ:** The default for standalone mode, providing efficient performance when data packets are small.
- **NATS:** A pure‑Go messaging solution that delegates memory management to Go’s garbage collector, currently supported only in standalone deployments.
- **Pulsar and Kafka:** These are available in cluster mode, offering better horizontal scalability for large‑scale applications.

For full configuration details and supported modes, refer to the Milvus documentation on [Message Storage with Milvus Operator](https://chatgpt.com/c/%EE%88%80cite%EE%88%8225%EE%88%81).

## The NATS Limitation: A Scalability Bottleneck

While NATS offers a streamlined, pure‑Go implementation, its use in Milvus is confined to standalone mode. This means that when a deployment requires scaling out across multiple nodes or clusters, NATS cannot participate in a distributed architecture. The reliance on Go’s garbage collector further complicates performance when handling larger data packets—specifically those above 64 KB, where response times could suffer.

This limitation is significant because horizontal scalability is one of the primary advantages that an on‑premises deployment should offer. Without it, organizations may face higher costs and operational overhead when trying to accommodate growing data and query loads.

## Comparing SaaS Vector Databases and On‑Premises Deployments

Many enterprises are drawn to on‑premises solutions for their data privacy, control, and cost‐optimization benefits. In contrast, popular SaaS vector databases like **Pinecone** offer seamless horizontal scaling, managed infrastructure, and reduced operational burden.

One of the key selling points of Milvus over SaaS alternatives has been the promise of local control and potential cost savings. However, when critical components such as message storage are limited—for example, NATS being available only in standalone mode—the scale-out advantages diminish. As a result, deploying Milvus at scale becomes not only technically challenging but also more expensive, as organizations must provision significantly larger infrastructures to handle the load.

## Exploring Alternatives: Pulsar, MinIO, and the Promise of Quadrant

Given these challenges, alternatives like Pulsar (which supports both standalone and cluster modes) and storage solutions like MinIO (for object storage) have been considered. Pulsar provides the horizontal scalability required for growing deployments, and MinIO offers cost‑effective storage solutions; however, integrating these components into an on‑premises Milvus setup can add layers of complexity.

This complexity has become a major driver behind our decision to transition to **Quadrant**. Despite its own set of challenges, Quadrant promises a more streamlined, scalable, and cost‑effective architecture that addresses the bottlenecks seen with NATS in Milvus. By moving to Quadrant, we expect to overcome the limitations inherent in a Milvus deployment that relies solely on NATS for messaging in standalone mode.

## Final Thoughts

While Milvus remains a powerful vector database for on‑premises applications, its current NATS compatibility confines deployments to standalone mode, hindering true horizontal scalability. This shortcoming, juxtaposed with the effortless scalability of SaaS solutions like Pinecone, makes large‑scale, on‑premises Milvus deployments both technically and financially challenging.

After evaluating alternative message storage solutions such as Pulsar and considering the benefits of integrated object storage with MinIO, our strategic decision is clear. To meet our scalability and cost‑efficiency goals, we are moving to Quadrant—a solution designed to overcome these hurdles while still offering the flexibility of an on‑premises deployment.
