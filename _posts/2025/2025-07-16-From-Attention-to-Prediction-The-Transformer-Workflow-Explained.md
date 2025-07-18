---
title: "From Attention to Prediction: The Transformer Workflow Explained"
date: "2025-07-16T06:31:30+00:00"
author: "gp"
layout: "post"
image: "/content/2025/07/transformer1.png"
categories:
  - "DNN"
  - "Transformer"
  - "LLM"
published: true
math: true
---


Transformer is a neural network architecture for sequence transduction that replaces recurrence and convolutions with **self‑attention**, enabling fully parallel processing of input embeddings \(X\in\mathbb{R}^{n\times d_{\text{model}}}\) via a **scaled dot‑product attention** mechanism for queries, keys, and values ([1], [2]). Introduced in “Attention Is All You Need,” it underpins models such as BERT, GPT, and T5 ([3], [4]).

The core building blocks are stacked encoder (and decoder) layers, each composed of:
1. **Multi‑Head Self‑Attention** that runs \(h\) parallel scaled‑dot‑product attentions and projects their concatenation back to the model dimension ([1], [3]).
2. **Position‑Wise Feed‑Forward Networks** (FFNs) that apply two linear transformations with a ReLU in between to each position independently ([4], [5]).
3. **Sinusoidal Positional Encodings** added to the input embeddings to inject order information ([6], [7]).
4. **Residual Connections** plus **Layer Normalization** around every sub‑layer to stabilize and accelerate training ([8], [9]).

---

## 1. Input Embeddings & Positional Encodings

Given token embeddings  
\[
X \in \mathbb{R}^{n\times d_{\text{model}}},
\]  
we add fixed sinusoidal positional encodings:  
\[
\mathrm{PE}(pos,2i) = \sin\!\Bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\Bigr),\quad
\mathrm{PE}(pos,2i+1) = \cos\!\Bigl(\frac{pos}{10000^{2i/d_{\text{model}}}}\Bigr),
\]  
where \(pos\) is the token index and \(i\) indexes the embedding dimension ([6], [10]). These encodings inject absolute and relative position information, allowing the model to be aware of token order without recurrence.

---

## 2. Self‑Attention & Scaled Dot‑Product

Project \(X\) into queries, keys, and values via learnable matrices \(W^Q, W^K, W^V\in\mathbb{R}^{d_{\text{model}}\times d_k}\):  
\[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V.
\]  
Compute attention scores and weighted sum:  
\[
\mathrm{Attention}(Q,K,V)
= \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)\,V,
\]  
where dividing by \(\sqrt{d_k}\) prevents overly large dot‑products and stabilizes gradients through the softmax ([2], [11]).

---

## 3. Multi‑Head Attention

Instead of a single attention, run \(h\) heads in parallel, each with its own projections \(W_i^Q, W_i^K, W_i^V\):  
\[
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\;KW_i^K,\;VW_i^V),
\qquad
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\;W^O,
\]  
where \(W^O\in\mathbb{R}^{h\,d_v\times d_{\text{model}}}\) projects the concatenated outputs back to the model dimensionality ([1], [3]).

---

## 4. Position‑Wise Feed‑Forward Network

After attention, each position’s vector \(x\in\mathbb{R}^{d_{\text{model}}}\) is transformed by a two‑layer MLP with ReLU activation:  
\[
\mathrm{FFN}(x) = \bigl(\max(0,\,xW_1 + b_1)\bigr)\,W_2 + b_2,
\]  
where \(W_1\in\mathbb{R}^{d_{\text{model}}\times d_{ff}}\), \(W_2\in\mathbb{R}^{d_{ff}\times d_{\text{model}}}\), and typically \(d_{ff}\gg d_{\text{model}}\) ([4], [5]).

---

## 5. Residual Connections & Layer Normalization

Each sub‑layer \(\mathcal{S}\) (self‑attention or FFN) is wrapped as follows:  
\[
y = x + \mathcal{S}(x),\quad
\mathrm{LayerNorm}(y) = \gamma\,\frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\]  
where \(\mu\) and \(\sigma^2\) are the mean and variance over the feature dimension of \(y\), and \(\gamma\), \(\beta\) are learned scale and shift parameters ([8], [9]). This pattern stabilizes gradient flow and accelerates convergence.

---

## 6. Complete Transformer Layer

Putting it all together, one **encoder** layer performs:

1. \(x' = \mathrm{LayerNorm}\bigl(x + \mathrm{MultiHead}(x,x,x)\bigr)\)  
2. \(\mathrm{LayerNorm}\bigl(x' + \mathrm{FFN}(x')\bigr)\)

Stack \(L\) such layers for the encoder. The **decoder** adds an additional encoder–decoder attention sub‑layer between self‑attention and FFN, following the same residual + normalization pattern.

---

## References

[1]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf  
[2]: https://medium.com/@funcry/in-depth-understanding-of-attention-mechanism-part-ii-scaled-dot-product-attention-and-its-7743804e610e  
[3]: https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism  
[4]: https://medium.com/image-processing-with-python/the-feedforward-network-ffn-in-the-transformer-model-6bb6e0ff18db  
[5]: https://www.tutorialspoint.com/gen-ai/feed-forward-neural-network-in-transformers.htm  
[6]: https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model  
[7]: https://www.machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1  
[8]: https://medium.com/@punya8147_26846/layer-normalization-and-residual-connections-in-transformer-layers-f7ed9a96a1ae  
[9]: https://arxiv.org/pdf/2002.04745  
[10]: https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers  
[11]: https://apxml.com/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/scaled-dot-product-attention  

