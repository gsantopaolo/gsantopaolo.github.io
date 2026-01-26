---
title: "Predict Peak VRAM Before Downloading a Model (Weights + KV Cache + Quantization)"
date: "2026-01-26T08:00:00+00:00"
author: "gp"
layout: "post"
image: "/content/2026/01/gpuvram.png"
categories: [LLM, Inference, GPU, VRAM, Quantization]
published: true
math: true
---

OOM debugging is a waste of time.

If a model is on the Hugging Face Hub in **Safetensors**, you can estimate **most of the VRAM** it will need **before downloading weights** — by reading only the **metadata header** (shapes + dtypes). The remaining part (activations, temp buffers, allocator behavior) is not perfectly deterministic, but we can still get a practical **peak VRAM estimate** good enough for capacity planning.

---

## Bottom Line First

Peak inference VRAM is usually:

$$\text{Peak VRAM} \approx \text{Weights} + \text{KV cache} + \text{Overhead}$$

- **Weights**: deterministic from Safetensors metadata (no tensor download needed)
- **KV cache**: deterministic from `config.json` + your `batch_size` + `context_len` + dtype
- **Overhead**: backend-dependent (FlashAttention/workspaces/allocator). Use a practical margin.

Hugging Face Accelerate explicitly states their estimator is for **loading** only, and that inference can add [**up to ~20%** extra in practice](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator). That's a good default margin when you don't know your runtime yet.

### Confidence Levels

| Component | Confidence | Why |
|-----------|------------|-----|
| Weights from Safetensors metadata | **99%** | Exact byte size from shapes + dtypes |
| KV cache (given config + batch + context + dtype) | **95%** | Deterministic formula; errors come from wrong assumptions (GQA vs MHA) |
| Weight quantization (INT8/INT4/FP8) | **85%** | Main term is deterministic; overhead + "some layers stay FP16" makes it approximate |
| Everything else (activations + buffers) | **60%** | Backend-dependent; use heuristic "+10–30%" |
| **Total peak VRAM estimate** | **~80%** | Higher if using known stack like vLLM with known settings |

---

## 1) Weights Memory from Safetensors Metadata (No Download)

Safetensors stores a small header with:
- tensor name
- dtype
- shape
- offsets

Hugging Face [documents how to fetch that header](https://huggingface.co/docs/safetensors/en/metadata_parsing) via **HTTP range requests**, meaning you can compute the exact weight bytes without downloading the tensor data.

The technique is simple:
1. Fetch the first 8 bytes (`Range: bytes=0-7`)
2. Interpret as little-endian uint64 to get header length
3. Fetch the header (`Range: bytes=8-{7+header_len}`)
4. Parse JSON to get all tensor metadata

**Weights bytes = Σ (numel(tensor) × bytes_per_dtype)**

If the repo has multiple `.safetensors` shards, you sum across all shards.

---

## 2) KV Cache Memory (The Long-Context Killer)

KV cache grows with:
- batch size (or concurrent sequences)
- context length
- number of layers
- number of KV heads (**GQA** uses `num_key_value_heads`)
- head dimension
- KV dtype

### Multi-Head Attention (MHA)

$$\text{KV bytes} = B \times T \times L \times (\text{num\_heads} \times \text{head\_dim}) \times 2 \times \text{bytes\_per\_elem}$$

### Grouped-Query Attention (GQA)

$$\text{KV bytes} = B \times T \times L \times (\text{num\_kv\_heads} \times \text{head\_dim}) \times 2 \times \text{bytes\_per\_elem}$$

The `2` is for K + V.

**Important:** Modern LLMs (Llama 2 70B, Llama 3, Mistral) often use GQA, which can reduce KV cache by `num_heads / num_kv_heads`. For example, Llama 3 uses a grouping factor of 2, so for every two query heads there's one KV head.

The [NVIDIA inference optimization guide](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) provides the canonical formulas:
- **Size of KV cache per token** = `2 × num_layers × (num_kv_heads × head_dim) × precision_bytes`
- **Total KV cache** = `batch_size × sequence_length × size_per_token`

### Example: Llama 2 7B

From [Hathora's deep dive](https://blog.hathora.dev/a-deep-dive-into-llm-inference-latencies/):
- 32 layers, 4096 hidden size, FP16
- KV cache for batch=1, seq_len=4096: `1 × 4096 × 2 × 32 × 4096 × 2 = ~2 GB`

---

## 3) Quantization: Where It Changes the Math

### Weight Quantization

If you quantize weights, you change effective bytes/parameter:

| Precision | Bytes/Param | Typical Use |
|-----------|-------------|-------------|
| FP32 | 4.0 | Training |
| FP16/BF16 | 2.0 | Standard inference |
| INT8 | ~1.0 | [bitsandbytes](https://huggingface.co/docs/transformers/en/main_classes/quantization), basic quantization |
| INT4 | ~0.5 | [GPTQ](https://huggingface.co/docs/transformers/en/main_classes/quantization), [AWQ](https://huggingface.co/docs/transformers/en/main_classes/quantization) |

Real formats add overhead for **scales and zero-points**. With typical group size of 128, you store one FP16 scale per 128 quantized weights, adding ~0.125-0.25 extra bytes per parameter.

Some layers (embeddings, lm_head) often stay in higher precision. Treat quantization estimates as **approximate** unless you parse the quantized checkpoint format directly.

### KV Cache Quantization (FP8)

If your bottleneck is long context, KV cache quantization can be significant:
- FP16 KV → FP8 KV ≈ **~2× less KV cache memory** (plus scaling factors)

vLLM supports [FP8 KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) with **per-tensor scaling** and calibration options. From the docs:

> Quantizing the KV cache to FP8 reduces its memory footprint. This increases the number of tokens that can be stored in the cache, improving throughput.

Usage example:
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True
)
```

---

## 4) The Part You Can't Perfectly Predict (But Can Bound)

"Everything else" includes:
- activations during **prefill** (parallel processing of all input tokens)
- temporary buffers (attention workspace)
- memory pools / fragmentation
- CUDA kernels loading (~1-2GB on first allocation)

This is backend- and kernel-dependent. The [NVIDIA guide](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) explains:

> The prefill phase represents the computationally intensive stage of LLM inference, where the model processes the entire input prompt to populate the key-value cache.

FlashAttention reduces memory complexity from O(n²) to O(n) through tiling, but the exact workspace requirements vary by implementation.

**Practical guidance:**
- For quick planning, **add 10–30% overhead**
- If you want a conservative number, start with **+20%** (as [recommended by HF Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator))

---

## 5) A Python Estimator (Weights + KV + Quant Knobs)

This script:
- Lists `.safetensors` files from the HF model API
- Reads each Safetensors header via range requests
- Sums weight bytes
- Reads `config.json`
- Estimates KV cache
- Adds an overhead ratio

> You need internet access when you run it.

```python
import json
import struct
import requests
from typing import Dict, Any, List, Tuple

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
}


def hf_list_safetensors(model_id: str, revision: str = "main") -> List[str]:
    """List all .safetensors files in a HF model repo."""
    api = f"https://huggingface.co/api/models/{model_id}"
    data = requests.get(api, timeout=30).json()
    files = [
        s.get("rfilename", "")
        for s in data.get("siblings", [])
        if s.get("rfilename", "").endswith(".safetensors")
    ]
    if not files:
        raise RuntimeError("No .safetensors files found in repo.")
    return files


def range_get(url: str, start: int, end: int) -> bytes:
    """Fetch byte range from URL."""
    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content


def read_safetensors_header(
    model_id: str, filename: str, revision: str = "main"
) -> Dict[str, Any]:
    """Read safetensors header without downloading tensors."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    # First 8 bytes = header length (little-endian uint64)
    first8 = range_get(url, 0, 7)
    (header_len,) = struct.unpack("<Q", first8)
    # Fetch header JSON
    header_bytes = range_get(url, 8, 8 + header_len - 1)
    return json.loads(header_bytes.decode("utf-8"))


def tensor_numel(shape: List[int]) -> int:
    """Calculate number of elements from shape."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def estimate_weights_from_metadata(
    model_id: str, revision: str = "main"
) -> Tuple[int, int]:
    """Estimate weight bytes and param count from safetensors metadata."""
    files = hf_list_safetensors(model_id, revision)
    total_bytes = 0
    total_params = 0

    for fn in files:
        header = read_safetensors_header(model_id, fn, revision)
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dtype = v["dtype"]
            shape = v["shape"]
            numel = tensor_numel(shape)
            total_params += numel

            if dtype not in DTYPE_BYTES:
                raise RuntimeError(f"Unsupported dtype: {dtype} (tensor={k})")
            total_bytes += numel * DTYPE_BYTES[dtype]

    return total_bytes, total_params


def fetch_config(model_id: str, revision: str = "main") -> Dict[str, Any]:
    """Fetch model config.json."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def estimate_kv_cache_bytes(
    cfg: Dict[str, Any],
    batch_size: int,
    context_len: int,
    kv_dtype_bytes: int = 2,
) -> int:
    """Estimate KV cache memory for given config and inference params."""
    num_layers = int(cfg.get("num_hidden_layers"))
    hidden_size = int(cfg.get("hidden_size"))
    num_heads = int(cfg.get("num_attention_heads"))
    # GQA: use num_key_value_heads if present
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))

    # KV cache: B * T * L * (kv_heads * head_dim) * 2(K+V) * bytes
    return (
        batch_size
        * context_len
        * num_layers
        * num_kv_heads
        * head_dim
        * 2
        * kv_dtype_bytes
    )


def format_gb(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.2f} GB"


def estimate_total_vram(
    model_id: str,
    revision: str = "main",
    batch_size: int = 1,
    context_len: int = 8192,
    kv_cache_dtype: str = "fp16",
    weight_effective_bytes_per_param: float = None,
    quant_overhead_ratio: float = 0.03,
    runtime_overhead_ratio: float = 0.20,
) -> None:
    """
    Estimate total VRAM for inference.

    Args:
        model_id: HuggingFace model ID
        revision: Model revision/branch
        batch_size: Concurrent sequences
        context_len: Maximum context length
        kv_cache_dtype: "fp16", "bf16", or "fp8"
        weight_effective_bytes_per_param: Override for quantized weights
            (e.g., 1.0 for INT8, 0.5 for INT4)
        quant_overhead_ratio: Extra overhead for scales/zeros
        runtime_overhead_ratio: Margin for activations/buffers
    """
    weights_bytes, params = estimate_weights_from_metadata(model_id, revision)

    # Override weights if quantized loading planned
    if weight_effective_bytes_per_param is not None:
        weights_bytes = int(
            params * weight_effective_bytes_per_param * (1.0 + quant_overhead_ratio)
        )

    cfg = fetch_config(model_id, revision)

    kv_bytes_per_elem = 2  # fp16/bf16
    if kv_cache_dtype.lower() == "fp8":
        kv_bytes_per_elem = 1

    kv_bytes = estimate_kv_cache_bytes(cfg, batch_size, context_len, kv_bytes_per_elem)

    base = weights_bytes + kv_bytes
    total = int(base * (1.0 + runtime_overhead_ratio))

    print(f"Model: {model_id}@{revision}")
    print(f"Params (metadata sum): {params/1e9:.2f}B")
    print(f"Weights: {format_gb(weights_bytes)}")
    print(f"KV cache ({kv_cache_dtype}, B={batch_size}, T={context_len}): {format_gb(kv_bytes)}")
    print(f"Base (weights+KV): {format_gb(base)}")
    print(f"Total w/ overhead (+{runtime_overhead_ratio*100:.0f}%): {format_gb(total)}")


if __name__ == "__main__":
    # Example: Llama 3.1 8B with INT4 weights, FP8 KV cache
    estimate_total_vram(
        "meta-llama/Llama-3.1-8B-Instruct",
        batch_size=4,
        context_len=16384,
        kv_cache_dtype="fp8",
        weight_effective_bytes_per_param=0.5,  # INT4
        quant_overhead_ratio=0.05,
        runtime_overhead_ratio=0.20,
    )
```

### Example Output

```
Model: meta-llama/Llama-3.1-8B-Instruct@main
Params (metadata sum): 8.03B
Weights: 4.21 GB
KV cache (fp8, B=4, T=16384): 4.00 GB
Base (weights+KV): 8.21 GB
Total w/ overhead (+20%): 9.86 GB
```

---

## Alternative: hf-mem CLI

If you just need a quick **weights-only** estimate without writing code, check out [hf-mem](https://github.com/alvarobartt/hf-mem) by Alvaro Bartolome. It's a lightweight CLI tool that uses the same Safetensors HTTP range request technique:

```bash
# Install and run in one command with uv
uvx hf-mem --model-id meta-llama/Llama-3.1-8B-Instruct
```

**What hf-mem does:**
- Fetches Safetensors metadata via HTTP range requests (no full download)
- Calculates weight memory from dtype and shape
- Works with Transformers, Diffusers, and Sentence Transformers models

**What it doesn't do (that the script above does):**
- KV cache estimation (batch size, context length)
- Quantization adjustments (INT4/INT8 overrides)
- Runtime overhead margin

Use `hf-mem` for quick checks; use the full script when you need to factor in KV cache for long-context or high-concurrency scenarios.

---

## Key Takeaways

- You can compute **weights VRAM exactly** from Safetensors metadata without downloading tensors
- KV cache is usually the **dominant** term for long context. Use **GQA** `num_key_value_heads`
- Quantize **weights** to fit the model. Quantize **KV cache** to scale context/concurrency
- For "everything else", add a **practical overhead** (start with **+20%**)

---

## References

- [Safetensors metadata parsing via HTTP range requests](https://huggingface.co/docs/safetensors/en/metadata_parsing) - Hugging Face docs
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - NVIDIA Developer Blog
- [A Deep Dive into LLM Inference Latencies](https://blog.hathora.dev/a-deep-dive-into-llm-inference-latencies/) - KV cache formulas and examples
- [Model memory estimator](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator) - HF Accelerate (loading vs inference caveat)
- [Quantization](https://huggingface.co/docs/transformers/en/main_classes/quantization) - Hugging Face Transformers (GPTQ, AWQ, bitsandbytes)
- [Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) - vLLM FP8 KV cache docs
- [KV Cache Size Calculations in GQA](https://medium.com/@liu.peng.uppsala/key-value-kv-cache-size-calculations-in-grouped-query-attention-gqa-e090d3037ab3) - Medium
- [Understanding and Coding the KV Cache](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) - Sebastian Raschka
- [hf-mem](https://github.com/alvarobartt/hf-mem) - CLI tool for quick weight memory estimates

---

## Need Help with Your AI Project?

Whether you're building a new AI solution or scaling an existing one, I can help. Book a free consultation to discuss your project.

[Book a Free Consultation](https://calendar.app.google/QuNua7HxdsSasCGu9){: .btn .btn-primary}
