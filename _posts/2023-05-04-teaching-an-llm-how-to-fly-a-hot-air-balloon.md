---
id: 341
title: 'Teaching an LLM How to Fly a Hot Air Balloon'
date: '2023-05-04T11:52:12+00:00'
author: gp
layout: post
guid: 'https://genmind.ch/?p=341'
permalink: /teaching-an-llm-how-to-fly-a-hot-air-balloon/
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
image: /content/2025/03/ballon2.jpg
categories:
    - 'Fine Tuning'
    - LLM
tags:
    - fine-tuning
    - trl
---

Welcome to this exciting journey where we merge machine learning with the skies! In this post, we’ll show you how we created a dataset from the FAA manual to fly a hot air balloon—and now it’s time to teach our AI to fly one. We’ll walk through the key ideas and how they’re implemented in our code, mixing concepts with the code itself for a fun, hands-on explanation.

---

## Merging the Manual with Machine Learning

Our goal is simple: fine-tune a large language model so that it can answer questions based on the FAA manual for flying a hot air balloon. The dataset, carefully curated from the manual, contains questions and answers that detail everything from pre-flight checks to emergency procedures. With this data in hand, we’re ready to teach our AI to become an expert balloon pilot!

### Fine-Tuning with TRL and LoRA

To efficiently adapt our pre-trained model (in our case, Qwen1.5-7B-Chat) to the specialized domain of hot air balloon flight, we use a combination of:

- **TRL (Transformer Reinforcement Learning):** The SFTTrainer from TRL helps us with supervised fine-tuning.
- **LoRA (Low-Rank Adaptation):** Instead of retraining the entire model, we use LoRA to update only a small fraction of the parameters. This keeps our training resource-friendly while still achieving impressive results.

Let’s take a look at how these concepts are implemented in the code.

---

## The Code: From Setup to Flight Training

### 1. Setting Up the Environment and Configuration

Before training, we set up our environment. This includes loading our FAA hot air balloon dataset and configuring the model and LoRA settings. Notice how we define the model details, cache directory, and fine-tuning tag to track our experiment.

```

model_id = "Qwen/Qwen1.5-7B-Chat"
fine_tune_tag = "faa-balloon-flying-handbook"
cache_dir = "cache"
upload_to_hf = True
```

Here, we also prepare our LoRA configuration to focus on key model modules, ensuring that our training is both efficient and effective:

```

peft_config = LoraConfig(
    r=16,
    modules_to_save=["lm_head", "embed_tokens"],
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

```

### 2. Crafting the Conversation from the FAA Manual

The heart of our dataset is a conversation template that incorporates a system message with detailed instructions from the FAA manual. This guides the AI to generate responses that reflect the practicalities and safety guidelines of hot air balloon flight.

```

system_message = """answer the given balloon flying handbook question by providing a clear, detailed explanation that references guidance from the balloon flying handbook, operational procedures, and relevant flight concepts.

provide a detailed breakdown of your answer, beginning with an explanation of the question and its context within the balloon flying handbook, followed by step-by-step reasoning based on the information provided in the handbook and applicable flight operation procedures. use logical steps that build upon one another to arrive at a comprehensive solution.

# steps

1. **understand the question**: restate the given question and clearly identify the main query along with any relevant details about balloon operations, safety procedures, or flight scenarios as discussed in the balloon flying handbook.
2. **handbook context**: explain the relevant procedures and guidelines as outlined in the balloon flying handbook. reference specific sections of the handbook, such as pre-flight checks, flight planning, emergency procedures, and operational parameters central to the question.
3. **detailed explanation**: provide a step-by-step breakdown of your answer. describe how you arrived at each conclusion by citing pertinent sections of the handbook and relevant operational standards.
4. **double check**: verify that your explanation is consistent with the guidelines in the balloon flying handbook and accurate according to current practices. mention any alternative methods or considerations if applicable.
5. **final answer**: summarize your answer clearly and concisely, ensuring that it is accurate and fully addresses the question.

# notes

- clearly define any terms or procedures specific to balloon flight operations as described in the handbook.
- include relevant procedural steps, operational parameters, or safety guidelines where applicable to support your answer.
- assume a familiarity with basic flight operation concepts while avoiding overly technical jargon unless it is commonly used in the ballooning community.
"""

```

We then create a function to format each sample from our dataset into a conversation structure that our model understands:

```

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }

```

### 3. Loading, Tokenizing, and Training

We load our model and tokenizer with a configuration optimized for memory efficiency. This includes 4-bit quantization, which is essential when working with large models.

```

def load_model_and_tokenizer(model_id, cache_dir):
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=True,
        cache_dir=cache_dir,
    )
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

```

After loading the data and applying our conversation template, we tokenize the conversations so the model can process them. The tokenization function joins all messages into a single string, ensuring we respect the maximum sequence length:

```

def tokenize(sample):
    conversation_strs = []
    for conversation in sample["messages"]:
        conv_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        conversation_strs.append(conv_str)
    tokenized = tokenizer(conversation_strs, truncation=True, max_length=1024)
    tokenized["text"] = conversation_strs
    return tokenized

```

Finally, our training process kicks off with the SFTTrainer. We set parameters like learning rate, batch size, and evaluation strategy to balance performance and resource constraints. The training arguments ensure we can monitor progress and save checkpoints at regular intervals.

```

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    args=TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=4e-5,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        save_steps=50,
        bf16=True,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        do_eval=True,
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=5,
        report_to=["tensorboard"],
        save_strategy="epoch",
        seed=42,
        output_dir=save_dir,
        log_level="debug",
    ),
)

```

With this setup, our AI is now learning the ins and outs of flying a hot air balloon as per the FAA manual guidelines. The training wraps up with saving our fine-tuned model, and if desired, uploading it to the Hugging Face Hub for broader access.

---

## Ready for Liftoff!

Now that we have created the dataset from the FAA manual, it’s time to teach our AI to fly a hot air balloon! By blending state-of-the-art fine-tuning techniques with our unique dataset, we’re giving our AI the tools to answer complex questions about hot air balloon operations with confidence.

Check out the full source code on my [GitHub repository](https://github.com/gsantopaolo/fine-tuning) and feel free to contribute or ask questions. Happy flying and fine tuning!
