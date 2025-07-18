---
title: 'From Attention to Prediction: The Transformer Workflow Explained'
date: '2025-07-16T6:31:30+00:00'
author: gp
layout: post
image: /content/2025/07/transformer1.png
categories: [DNN,  Transformer, LLM]
published: true
math: true
---
## Summary

The Transformer replaces recurrence with **self‑attention**, where input embeddings $X\in\mathbb{R}^{n\times d_{\text{model}}}$ are projected into **queries**, **keys**, and **values** and combined via a **scaled dot‑product attention** mechanism to capture contextual relationships in parallel ([NeurIPS Papers][1], [Medium][2]). Multiple such attentions run in parallel as **Multi‑Head Attention**, concatenating $h$ heads and projecting back to $d_{\text{model}}$ for richer representation subspaces ([NeurIPS Papers][1], [GeeksforGeeks][3]). Each layer adds a **position‑wise feed‑forward network**, consisting of two linear transformations with a ReLU activation in between, to introduce non‑linearity at each position independently ([Medium][4], [TutorialsPoint][5]). To provide sequence order information without recurrence, **sinusoidal positional encodings** are added to the embeddings before any attention is applied ([Data Science Stack Exchange][6], [MachineLearningMastery.com][7]). Finally, every sub‑layer is wrapped in a **residual connection** followed by **Layer Normalization**, which stabilizes training by normalizing features to zero mean and unit variance then scaling and shifting via learned parameters ([Medium][8], [arXiv][9]).

---

## 1. Input Embeddings & Positional Encodings

Given token embeddings $X \in \mathbb{R}^{n\times d_{\text{model}}}$, fixed **sinusoidal positional encodings** $\mathrm{PE}\in\mathbb{R}^{n\times d_{\text{model}}}$ are added to inject order:

$$
\mathrm{PE}(pos,2i) = \sin\!\Bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\Bigr),\quad
\mathrm{PE}(pos,2i+1) = \cos\!\Bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\Bigr),
$$

where $pos$ is the token position and $i$ indexes the dimension ([Data Science Stack Exchange][6], [GeeksforGeeks][10]).

---

## 2. Scaled Dot‑Product Attention

For queries $Q$, keys $K$, and values $V$ (all $\in\mathbb{R}^{n\times d_k}$):

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr)\,V,
$$

where the $\tfrac{1}{\sqrt{d_k}}$ scaling counteracts large dot‑product magnitudes, stabilizing gradients through the softmax ([NeurIPS Papers][1], [ApX Machine Learning][11]).

---

## 3. Multi‑Head Attention

With $h$ attention heads, each head $i$ uses separate projection matrices $W_i^Q,W_i^K,W_i^V\in\mathbb{R}^{d_{\text{model}}\times d_k}$:

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\;KW_i^K,\;VW_i^V),\quad
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\;W^O,
$$

with $W^O\in\mathbb{R}^{hd_v\times d_{\text{model}}}$ projecting back to $d_{\text{model}}$ ([NeurIPS Papers][1], [GeeksforGeeks][3]).

---

## 4. Position‑Wise Feed‑Forward Network

Applied to each position independently, the **FFN** is:

$$
\mathrm{FFN}(x) = \bigl(\max(0,\,xW_1 + b_1)\bigr)\,W_2 + b_2,
$$

where $W_1\in\mathbb{R}^{d_{\text{model}}\times d_{ff}}$, $W_2\in\mathbb{R}^{d_{ff}\times d_{\text{model}}}$, and $d_{ff}\gg d_{\text{model}}$ (e.g.\ 2048 vs.\ 512) ([Medium][4], [TutorialsPoint][5]).

---

## 5. Residual Connections & Layer Normalization

Each sub‑layer $\mathcal{S}$ (attention or FFN) is wrapped as:

$$
\begin{aligned}
y &= x + \mathcal{S}(x),\\
\mathrm{LayerNorm}(y) &= \gamma \,\frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\end{aligned}
$$

where $\mu,\sigma^2$ are mean and variance across the feature dimension of $y$, and $\gamma,\beta$ are learned scale and shift parameters ([Medium][8], [arXiv][9]).

---

## 6. Complete Transformer Layer

Putting it together, one encoder (or decoder) layer computes:

1. **Self‑Attention sub‑layer**:
   $\text{norm}_1(x + \text{MultiHead}(xW^Q,xW^K,xW^V))$.
2. **Feed‑Forward sub‑layer**:
   $\text{norm}_2(y + \mathrm{FFN}(y))$.

Stacking $L$ such layers yields the full Transformer encoder (and similarly for the decoder, with added encoder–decoder attention).

[1]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf?utm_source=chatgpt.com "[PDF] Attention is All you Need - NIPS"
[2]: https://medium.com/%40funcry/in-depth-understanding-of-attention-mechanism-part-ii-scaled-dot-product-attention-and-its-7743804e610e?utm_source=chatgpt.com "In Depth Understanding of Attention Mechanism (Part II) - Scaled ..."
[3]: https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/?utm_source=chatgpt.com "Multi-Head Attention Mechanism - GeeksforGeeks"
[4]: https://medium.com/image-processing-with-python/the-feedforward-network-ffn-in-the-transformer-model-6bb6e0ff18db?utm_source=chatgpt.com "The Feedforward Network (FFN) in The Transformer Model - Medium"
[5]: https://www.tutorialspoint.com/gen-ai/feed-forward-neural-network-in-transformers.htm?utm_source=chatgpt.com "Feed Forward Neural Network in Transformers - Tutorialspoint"
[6]: https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model?utm_source=chatgpt.com "What is the positional encoding in the transformer model?"
[7]: https://www.machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/?utm_source=chatgpt.com "A Gentle Introduction to Positional Encoding in Transformer Models ..."
[8]: https://medium.com/%40punya8147_26846/layer-normalization-and-residual-connections-in-transformer-layers-f7ed9a96a1ae?utm_source=chatgpt.com "Layer Normalization and Residual Connections in Transformer Layers"
[9]: https://arxiv.org/pdf/2002.04745?utm_source=chatgpt.com "[PDF] On Layer Normalization in the Transformer Architecture - arXiv"
[10]: https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/?utm_source=chatgpt.com "Positional Encoding in Transformers - GeeksforGeeks"
[11]: https://apxml.com/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/scaled-dot-product-attention?utm_source=chatgpt.com "Scaled Dot-Product Attention Explained - ApX Machine Learning"
