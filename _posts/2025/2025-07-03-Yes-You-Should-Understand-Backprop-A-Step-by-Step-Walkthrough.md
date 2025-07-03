---
title: 'Yes, You Should Understand Backprop: A Step-by-Step Walkthrough'
date: '2025-07-03T6:31:30+00:00'
author: gp
layout: post
image: /content/2025/06/backprop.jpg
categories: [DNNs, Language Model, Backpropagation]
math: true
mermaid: true
published: true
---




Backpropagation—originating in Linnainmaa’s 1970 reverse-mode AD thesis and popularized for neural 
nets by Rumelhart et al. in 1986—is the workhorse that makes deep learning 
feasible ([en.wikipedia.org][1], [de.wikipedia.org][2]).  It efficiently computes gradients via a 
reverse traversal of a computation graph, applying the chain rule at each 
node ([colah.github.io][3], [cs231n.stanford.edu][4]).  
Understanding its mechanics by hand helps you debug and improve models when “autograd” 
breaks down—Karpathy warns that leaky abstractions hide vanishing/exploding gradients and 
dying ReLUs unless you look under the hood ([karpathy.medium.com][5], [cs231n.stanford.edu][6]).  
We’ll cover its history, intuition, vectorized formulas, a worked numeric example, common pitfalls, 
and modern remedies—with pointers to every key paper and lecture.

---

## Introduction

Backpropagation is the algorithm that computes the gradient of a scalar loss $L$ with respect 
to all network parameters by recursively applying the chain rule from outputs back to 
inputs ([colah.github.io][3], [cs231n.stanford.edu][4]).  
In code, frameworks like TensorFlow or PyTorch automate backward passes, but treating 
it as a black box can obscure subtle failure modes—debugging requires tracing error signals 
through each operation ([karpathy.medium.com][5]).

---

## A Brief History of Backpropagation

### 1. 1970: Reverse-Mode Automatic Differentiation

Seppo Linnainmaa’s 1970 M.Sc. thesis introduced “reverse-mode” AD: representing a composite function 
as a graph and computing its derivative by a backward sweep of local chain-rule 
applications ([en.wikipedia.org][1], [idsia.ch][7]).

### 2. 1974–1982: From AD to Neural Nets

Paul Werbos recognized that reverse-mode AD could train multi-layer perceptrons, presenting 
this idea in his 1974 Harvard PhD and later publications ([news.ycombinator.com][8]).

### 3. 1986: Rumelhart, Hinton & Williams

David Rumelhart, Geoff Hinton & Ronald Williams formalized backpropagation for neural 
networks in their landmark Nature paper, showing multi-layer nets could learn internal 
representations ([en.wikipedia.org][9]).

### 4. 2000s: The Deep Learning Boom

GPUs, large datasets, and architectural innovations (CNNs, RNNs) made deep nets practical. 
Backprop—once limited to shallow networks—now trains architectures with hundreds of 
layers ([colah.github.io][3], [jmlr.org][10]).

---

## Why You Must Understand Backprop by Hand

Even though autograd handles derivatives, Karpathy stresses that treating backprop as “magic” 
leads to **leaky abstractions**—you’ll miss why your model stalls, diverges, or suffers 
dead neurons ([karpathy.medium.com][5]).  
Engineers who can manually step through a backward sweep catch vanishing/exploding gradients 
and activation pitfalls early, saving days of debugging.

---

## How Backpropagation Actually Works

### A. Computation Graphs & the Chain Rule

Every operation in a network is a node in a directed acyclic graph. Given $y = f(x)$ and a 
loss $L$, the backward step computes

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial y}\;\frac{\partial y}{\partial x},
$$

re-using $\partial L/\partial y$ as the “upstream” signal at each 
edge ([colah.github.io][3], [cs231n.stanford.edu][4]).

![simple chain-rule graph](/content/2025/06/tree-eval-derivs.png){: width="500" height="300" }
_simple chain-rule graph, source: [Calculus on Computational Graphs: Backpropagation][3]_

### B. Vectorized Layer-Wise Updates

For a fully-connected layer $l$ with inputs $a^{l-1}$, pre-activations $z^l=W^l a^{l-1}+b^l$, 
activations $a^l=f(z^l)$, and loss gradient $\delta^l = \partial L/\partial z^l$, the updates are:

$$
\nabla_{W^l}L = \delta^l\, (a^{l-1})^T,\quad
\nabla_{b^l}L = \sum_i \delta^l_i,\quad
\delta^{l-1} = (W^l)^T\,\delta^l \;\circ\; f'(z^{l-1})
$$

where $\circ$ is element-wise multiplication ([cs231n.stanford.edu][4], [cs231n.github.io][11]).


![vectorized backprop](/content/2025/06/vectorized-operations.png){: width="500" height="300" }
_vectorized backprop, source: [CS231N slide 53][https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf]_


### C. Worked Numerical Example

Matt Mazur’s two-layer network example crunches actual numbers—forward activations to (0.01, 0.99) 
and backward gradients to concrete $\Delta W$, $\Delta b$ updates. Stepping through 
it cements intuition ([colah.github.io][3], [web.stanford.edu][12]).

---

## Pitfalls & Practical Tips

### 1. Vanishing & Exploding Gradients

Sigmoid/tanh saturation ($f'(z)\to0$) shrinks gradients exponentially in depth; poor 
initialization can blow them up. Use Xavier/He initialization and, for RNNs, 
gradient clipping ([karpathy.medium.com][5], [cs231n.stanford.edu][6]).


![Sigmoid derivative vanishing plot](/content/2025/06/sigmoid-derivative-vanishing-plot.png){: width="500" height="300" }
_Sigmoid derivative vanishing plot, source: [karpathy.medium.com][5]_

### 2. Dying ReLUs

ReLUs zero-out for negative inputs; if a neuron’s gradient path stays negative, it “dies.” 
Karpathy calls this “brain damage.” Leaky-ReLU, PReLU or ELU keep a small slope 
for $z<0$ ([karpathy.medium.com][5], [substack.com][13]).


![Sigmoid derivative vanishing plot](/content/2025/06/dying-ReLUs.png){: width="500" height="300" }
_ReLU zero-slope region, source: [karpathy.medium.com][5]_

### 3. Modern Remedies

Batch normalization, residual/skip connections (ResNets), and newer activations (SELU) 
preserve signal flow in very deep nets ([karpathy.medium.com][5], [jmlr.org][10]).

---

## Conclusion & Further Reading

A hands-on grasp of backpropagation is essential for debugging, architecture design, and research 
innovation. For deeper dives, see:

* Collobert & Weston, “Natural Language Processing (Almost) from Scratch” (JMLR 2011) ([jmlr.org][10])
* CS231n “Derivatives, Backpropagation, and Vectorization” handout ([cs231n.stanford.edu][4])
* Nielsen, *Neural Networks and Deep Learning*, Ch. 2
* Olah, “Calculus on Computational Graphs: Backpropagation” ([colah.github.io][3])
* Inna Logunova, [Backpropagation in Neural Networks](https://serokell.io/blog/understanding-backpropagation?utm_source=genmind.ch)


[1]: https://en.wikipedia.org/wiki/Seppo_Linnainmaa?utm_source=genmind.ch "Seppo Linnainmaa"
[2]: https://de.wikipedia.org/wiki/Backpropagation?utm_source=genmind.ch "Backpropagation"
[3]: https://colah.github.io/posts/2015-08-Backprop/?utm_source=genmind.ch "Calculus on Computational Graphs: Backpropagation - colah's blog"
[4]: https://cs231n.stanford.edu/handouts/derivatives.pdf?utm_source=genmind.ch "[PDF] Derivatives, Backpropagation, and Vectorization - CS231n"
[5]: https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b?utm_source=genmind.ch "Yes you should understand backprop | by Andrej Karpathy | Medium"
[6]: https://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf?utm_source=genmind.ch "[PDF] Backpropagation and Gradients - CS231n"
[7]: https://www.idsia.ch/~juergen/who-invented-backpropagation.html?utm_source=genmind.ch "Who Invented Backpropagation? - IDSIA"
[8]: https://news.ycombinator.com/item?id=35479272&utm_source=genmind.ch "Seppo Linnainmaa, first publisher of \"reverse mode of automatic ..."
[9]: https://en.wikipedia.org/wiki/Backpropagation?utm_source=genmind.ch "Backpropagation"
[10]: https://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf?utm_source=genmind.ch "[PDF] Natural Language Processing (Almost) from Scratch"
[11]: https://cs231n.github.io/optimization-2/?utm_source=genmind.ch "Backpropagation - CS231n Deep Learning for Computer Vision"
[12]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/slides/cs224n-2021-lecture03-neuralnets.pdf?utm_source=genmind.ch "[PDF] Neural net learning: Gradients by hand (matrix calculus) and ..."
[13]: https://substack.com/home/post/p-163881360?utm_campaign=post&utm_medium=web&utm_source=genmind.ch "Andrej Karpathy is right, you should understand backprop, in Java."
