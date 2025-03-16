---
id: 361
title: 'The Best OSS Style Transfer Implementation I&#8217;ve Tried'
date: '2022-10-14T18:22:22+00:00'
author: 'Gian Paolo'
layout: post
guid: 'https://genmind.ch/?p=361'
permalink: /the-best-oss-style-transfer-implementation-ive-tried/
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
image: /content/2025/03/golden-starry.jpg
categories:
    - 'Deep Learning'
    - 'Machine Learning'
tags:
    - 'style transfer'
---

I’ve was on the hunt for an open-source style transfer implementation able to produce the output I saw on the different papers, after trying several implementations and getting underwhelming outputs, I finally came across [Katherine Crowson’s style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch). And wow—the results blew me away!

### What Makes This Implementation Stand Out

Katherine’s implementation based on the paper [“A Neural Algorithm of Artistic Style”](https://arxiv.org/abs/1508.06576) supports both CPUs and Nvidia GPUs, and even goes as far as producing high-resolution, print-ready stylizations. Some of the cool modifications from the original paper include:

- **Using PyTorch pre-trained VGG-19 weights:** A small change that makes a big difference.
- **Improved padding:** Changing the first layer’s padding mode to `'replicate'` helps reduce edge artifacts.
- **Scaled pooling results:** Ensuring the output magnitude remains consistent.
- **[Wasserstein-2 style loss](https://wandb.ai/johnowhitaker/style_loss_showdown/reports/An-Explanation-of-Style-Transfer-with-a-Showdown-of-Different-Techniques--VmlldzozMDIzNjg0#style-loss-#3:-%22vincent's-loss%22):** For a more refined style comparison.
- **Exponential moving average:** To reduce noise and improve the overall output.
- **Multi-scale stylization:** Stylizing images at progressively larger scales (each larger by a factor of √2).

### Running on MPS (Mac) Without a Hitch

While testing on my Mac, I discovered that the code was defaulting to CPU—even though my system supports MPS. After a bit of digging, I found that in `cli.py` ([line 216](https://github.com/crowsonkb/style-transfer-pytorch/blob/master/style_transfer/cli.py)), the device selection wasn’t checking for MPS hardware. Here’s the quick fix I made:

XXXXXXXXXXXXX

With this small change, I was able to run the style transfer on my Mac’s GPU, and the performance was fantastic.

### The Results Speak for Themselves

For my first test, I applied the Starry Night style to a photo of the Golden Gate Bridge. The transformation was nothing short of extraordinary. (See the screenshot below, captured with asitop, showing the process running on GPU!)

![](content/2025/03/goldengate-300x169.jpg)

The image of the Golden Gate Bridge I used to apply the Starry Night style

![](content/2025/03/starry-night-300x225.jpg)

And the result? Check by yourself!

![](content/2025/03/golden-starry-300x169.jpg)

### Big Thanks

A huge shoutout to Katherine Crowson for her exceptional work on this project. If you’re into neural style transfer and want a reliable, high-quality implementation, I highly recommend giving this a try.

<del>As always, you can download the code here.</del> \[Edit\] I’ve pushed this fix to [my fork](https://github.com/gsantopaolo/style-transfer-pytorch)—check it out if you’re interested!

Happy styling!
