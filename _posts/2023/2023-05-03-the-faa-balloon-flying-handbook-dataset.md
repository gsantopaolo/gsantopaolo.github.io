---
title: 'The FAA Balloon Flying Handbook Dataset'
date: '2023-05-03T23:50:07+00:00'
author: gp
layout: post
image: /content/2025/03/balloons-cappadocia.jpg
categories:
  - dataset
---

I was searching for a dataset to fine-tune an LLM on a specific topic, but I couldn’t find one that met my needs. T This dataset is intended to help fine tune LLMs.

When I began fine-tuning language models for specific topic, I ran into a common problem: finding a high-quality, focused dataset on niche domains isn’t easy, that’s why I decided to create my own datase, and I come up with the idea of using the FAA Balloon Flying Handbook (FAA-H-8083-11B).

I started by downloading the official FAA Balloon Flying Handbook, which, thankfully, is available in PDF form. I used the convenient `pymupdf4llm` library to convert the PDFs into Markdown files.

Next, I split the Markdown files into manageable chunks, by analyzing the MDs I noticed that the best way to extract paragraphs was by using bold headings (`**`) as char splitter. Each section was cleanly separated and ready for further processing.

To generate usable question-answer pairs from these chunks, I crafted a reusable prompt and fed it through a language model, generating five QA pairs per chunk. These pairs captured the essence of each section in a structured way.

All these QA pairs, along with their respective context sections, were then combined into a DataFrame. For easy integration into various machine learning workflows, I saved the final DataFrame as a JSONL (JSON Lines) file, a format highly efficient for handling larger datasets.

\[edit\]

You can easily use the dataset [directly from HuggingFace](https://huggingface.co/datasets/gsantopaolo/faa-balloon-flying-handbook) like this:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("gsantopaolo/faa-balloon-flying-handbook")
print(dataset)

# Print the first 5 rows of the train split
for row in dataset["train"][:5]:
    print(row)
```

You can use the same approach to create your own datase, and if you need help, feel free to reach me out, all my contacts at the bottom of the [about page](https://genmind.ch/about/). As always, <del>you can download the full source code here</del> \[edit\] I moved all the code on my [GitHub repo](https://github.com/gsantopaolo/datasets/tree/main/balloon)

Happy coding!
