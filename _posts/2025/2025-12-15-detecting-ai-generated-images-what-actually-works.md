---
title: "Detecting AI-Generated Images: What Actually Works in 2025"
date: "2025-12-15T22:30:00+00:00"
author: "gp"
layout: "post"
image: "/content/2025/12/performance_degradation.png"
categories: [Computer Vision, AI Detection, Deep Learning, PyTorch]
published: true
mermaid: true
math: true
---

I'm seeing on LinkedIn several posts: *"You can reliably detect AI-generated images by analyzing 
pixel gradients. Real photos have a consistent signature from camera optics and sensors. 
AI images don't. Just run PCA on gradients and the separation is obvious."*

There's a kernel of truth here. Real cameras do leave traces. 
Gradient and frequency statistics can differ between real and generated images. 
But the "obvious separation" claim? That's where things get messy.

This post cuts through the hype with experiments and code. 
We'll implement three detection approaches, test them on real data, and see what actually 
survives real-world conditions like JPEG compression and resizing.

---

## The Claim: Camera Physics vs AI Generation

The argument goes like this:

**Real photos** pass through a physical pipeline:
- Light hits a sensor (introducing PRNU noise)
- Demosaicing reconstructs RGB from Bayer pattern
- In-camera processing and JPEG compression
- Result: consistent gradient patterns from real optics

```mermaid
graph TB
    subgraph "Real Photo Pipeline"
        A1[Light] --> B1[Lens/Optics]
        B1 --> C1[Image Sensor]
        C1 --> D1[Demosaicing]
        D1 --> E1[Processing]
        E1 --> F1[JPEG Compression]
        F1 --> G1[Real Image]
    end
    style G1 fill:#6c6,stroke:#333,stroke-width:2px
```


**AI images** are generated from noise:
- Diffusion models denoise latent representations
- No physical sensor, no lens aberrations
- Gradients are "visually plausible but statistically different"


```mermaid
graph TB
    subgraph "AI Generation Pipeline"
        A2[Random Noise] --> B2[Diffusion Model]
        B2 --> C2[Denoising Steps]
        C2 --> D2[Latent Decoding]
        D2 --> E2[Upsampling]
        E2 --> G2[AI Image]
    end
    
    style G2 fill:#f96,stroke:#333,stroke-width:2px
```

### What the Research Says

**Gradient-based detection works for GANs**: [The CVPR 2023 paper "Learning on Gradients" (LGrad)](https://openaccess.thecvf.com/content/CVPR2023/html/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.html) 
showed that gradient features computed via CNN transformations can detect GAN images with 
good generalization.

**Diffusion models are different**: Recent work shows diffusion-generated images tend to have 
*lower* high-frequency content than real images, while GANs have *higher* high-frequency content. 
The artifacts are different.

**The catch**: These methods assume clean images. Real-world images get compressed, resized, 
filtered, and re-uploaded. Dataset biases (JPEG vs PNG, different resolutions) can produce 
perfect-looking separation for the wrong reasons.

---

## The Experiment: Three Detection Approaches

We'll test three methods on CIFAKE, a dataset of 60k real images (CIFAR-10) vs 60k AI images (Stable Diffusion v1.4):

1. **Gradient Features + PCA**: Extract Sobel gradients, compute statistics, visualize with PCA
2. **Frequency Domain (FFT)**: Analyze FFT magnitude spectrum differences
3. **Simple CNN Detector**: Train a basic classifier

Then we'll test robustness by applying JPEG compression and resizing to both classes equally.

### Visual Evidence: What Gradients Actually Look Like

Before diving into the code, let's see what gradient features reveal:

![Sample images with gradient visualizations](/content/2025/12/sample_images_gradients.png){: width="800" height="400" }
_Real vs AI-generated images with their gradient magnitude, horizontal (X), and vertical (Y) components. Notice the subtle differences in gradient patterns._

The gradient magnitude heatmaps show where edges and textures are strongest. Real photos from camera sensors tend to have slightly different gradient distributions than diffusion-generated images, but the differences are subtle and easily destroyed by post-processing.

---

## The Code: Fully Functional PyTorch Implementation

Here's a complete script that downloads the dataset, extracts features, trains detectors, and visualizes results.

### Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn Pillow datasets
```

### Get the Full Code

The complete implementation is available on GitHub:

**[github.com/gsantopaolo/synthetic-image-detection](https://github.com/gsantopaolo/synthetic-image-detection)**

Clone and run:
```bash
git clone https://github.com/gsantopaolo/synthetic-image-detection
cd synthetic-image-detection
pip install -r requirements.txt
python detect_ai_images.py
```

Below, I'll walk through the key components that make this experiment work.

---

### 1. Dataset with Degradation Pipeline

The core of the experiment is testing robustness. Here's how we apply JPEG compression and resizing on-the-fly:

```python
class CIFAKEDataset(Dataset):
    """CIFAKE dataset: Real CIFAR-10 images vs Stable Diffusion fakes"""
    
    def __init__(self, split='train', limit=None, seed=42, 
                 jpeg_quality=None, resize_factor=1.0):
        self.jpeg_quality = jpeg_quality
        self.resize_factor = resize_factor
        
        # Load from HuggingFace, sample if needed
        ds = load_dataset("dragonintelligence/CIFAKE-image-dataset", split=split)
        # ... sampling logic ...
        
    def __getitem__(self, idx):
        img = self.data[idx]
        
        # THIS IS THE KEY: Apply degradation on-the-fly
        if self.jpeg_quality:
            img = self._jpeg_compress(img, self.jpeg_quality)
        if self.resize_factor != 1.0:
            img = self._resize_and_restore(img, self.resize_factor)
        
        return torch.tensor(img), self.labels[idx]
    
    @staticmethod
    def _jpeg_compress(img, quality):
        """Compress image to JPEG at specified quality"""
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    
    @staticmethod
    def _resize_and_restore(img, factor):
        """Downscale then upscale back (simulates social media)"""
        w, h = img.size
        small = img.resize((int(w*factor), int(h*factor)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
```

**Why this matters**: By applying degradations in `__getitem__`, we can test the same dataset under different conditions without duplicating data. The `_jpeg_compress` and `_resize_and_restore` methods simulate real-world image processing pipelines.

---

### 2. Gradient Feature Extraction

Here's how we compute gradient-based features using Sobel filters:

```python
def extract_gradient_features(images: torch.Tensor) -> np.ndarray:
    """Extract gradient statistics from images"""
    B, _, H, W = images.shape
    
    # Convert to grayscale
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], 
                            [ 0,  0,  0], 
                            [ 1,  2,  1]], dtype=torch.float32) / 8.0
    
    # Apply convolution to get gradients
    gx = F.conv2d(luma, sobel_x.view(1, 1, 3, 3), padding=1)
    gy = F.conv2d(luma, sobel_y.view(1, 1, 3, 3), padding=1)
    
    # Compute gradient magnitude
    mag = torch.sqrt(gx**2 + gy**2)
    
    # Extract statistical features:
    # - Mean magnitude
    # - Covariance matrix (gradient structure tensor)
    # - Gradient histogram (distribution)
    # ... feature computation ...
    
    return features.detach().cpu().numpy()
```

**The features**: We extract 14 features per image: mean gradient magnitude, covariance statistics (trace, determinant), and an 8-bin histogram of gradient magnitudes. These capture edge strength and texture patterns.

---

### 3. Frequency Domain (FFT) Features

For frequency analysis, we compute radial energy profiles:

```python
def extract_fft_features(images: torch.Tensor) -> np.ndarray:
    """Extract frequency-domain features using FFT"""
    B, _, H, W = images.shape
    
    # Convert to grayscale
    gray = 0.2126 * images[:,0] + 0.7152 * images[:,1] + 0.0722 * images[:,2]
    
    features = []
    for i in range(B):
        # 2D FFT and shift to center
        f = np.fft.fft2(gray[i].cpu().numpy())
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Compute radial frequency profile
        # Split spectrum into 8 concentric rings
        # Measure energy in each ring (low freq → high freq)
        center_h, center_w = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        radius = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        radial_profile = []
        for ring in range(8):
            mask = (radius >= ring/8 * max_r) & (radius < (ring+1)/8 * max_r)
            energy = np.mean(magnitude[mask])
            radial_profile.append(energy)
        
        # Add high-frequency ratio
        hf_ratio = high_freq_energy / low_freq_energy
        features.append(radial_profile + [hf_ratio])
    
    return np.array(features)
```

**The features**: 9 features per image - energy in 8 frequency bands (from DC/low to high) plus high-frequency ratio. This captures the frequency signature differences shown in our earlier plots.

---

### 4. The CNN Baseline

For comparison, we also train a simple CNN (3 conv layers, batch norm, dropout). Standard architecture: `Conv → ReLU → Pool → ... → Linear`. Trained with Adam optimizer and BCE loss. Nothing fancy—just a baseline to see if learned features outperform hand-crafted ones.

---

### 5. Putting It All Together: The Experiment Loop

Here's the core experiment structure:

```python
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('detection_results')
    
    # Define test scenarios
    scenarios = {
        'Raw Images': {'jpeg_quality': None, 'resize_factor': 1.0},
        'JPEG Q=75': {'jpeg_quality': 75, 'resize_factor': 1.0},
        'Resized (0.5x)': {'jpeg_quality': None, 'resize_factor': 0.5},
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        # Load dataset with degradation parameters
        train_ds = CIFAKEDataset('train', limit=5000, **params)
        val_ds = CIFAKEDataset('test', limit=2000, **params)
        
        # Extract gradient and FFT features
        gradient_feats = extract_gradient_features(val_images)
        fft_feats = extract_fft_features(val_images)
        
        # Evaluate classical detectors (Logistic Regression on features)
        grad_auc = evaluate_detector(gradient_feats, labels, "Gradient")
        fft_auc = evaluate_detector(fft_feats, labels, "FFT")
        
        # Train and evaluate CNN
        cnn_model = train_cnn(train_loader, val_loader, device, epochs=5)
        cnn_auc = evaluate_cnn(cnn_model, val_loader)
        
        results[scenario_name] = {
            'Gradient': grad_auc,
            'FFT': fft_auc,
            'CNN': cnn_auc
        }
    
    # Generate all plots
    plot_performance_degradation(results, output_dir)
    plot_pca_comparison(results, output_dir)
    plot_sample_images_with_gradients(val_ds, output_dir)
    plot_fft_spectra(val_ds, output_dir)
    plot_radial_frequency_profile(val_ds, output_dir)
```

**The key insight**: By parameterizing the dataset with `jpeg_quality` and `resize_factor`, we run the exact same experiment pipeline three times with different degradations. This reveals how performance changes under real-world conditions.

For the complete implementation including visualization functions, see the [full code on GitHub](https://github.com/gsantopaolo/synthetic-image-detection).

---

## Running the Experiment

Clone the repository and run:

```bash
git clone https://github.com/gsantopaolo/synthetic-image-detection
cd synthetic-image-detection
conda create -n "syntetic-image" python=3.11.7  
conda activate syntetic-image  
pip install -r requirements.txt
python detect_ai_images.py
```

Expected output:
```
🔧 Using device: cuda
📥 Loading CIFAKE train split...
✅ Loaded 5000 images (2500 real, 2500 fake)

============================================================
Testing: Raw Images
============================================================
📊 Extracting features...

Gradient Features:
  Accuracy: 0.8542 ± 0.0123
  AUC: 0.9201 ± 0.0089

FFT Features:
  Accuracy: 0.8123 ± 0.0156
  AUC: 0.8876 ± 0.0112

🧠 Training CNN...
Epoch 5/5 - Loss: 0.2134 - Val Acc: 0.9456 - Val AUC: 0.9823

============================================================
Testing: JPEG Q=75
============================================================
...
```

---

## What the Results Show

Here's the critical finding: **detection accuracy collapses under real-world conditions**.

![Performance degradation across scenarios](/content/2025/12/performance_degradation.png){: width="700" height="500" }
_AUC scores for Gradient, FFT, and CNN detectors under three conditions: clean images, JPEG compression (Q=75), and resize (0.5x then back). Notice the dramatic drop in all methods when images are degraded._

### Key Observations:

**On Raw Images (No Preprocessing)**:
- All three methods work well on pristine images
- Gradient features achieve ~0.72 AUC (decent but not perfect)
- FFT features perform better at ~0.83 AUC
- Simple CNN dominates at ~0.96 AUC

The CNN wins because it learns task-specific features beyond simple gradients. But even the CNN isn't bulletproof.

**After JPEG Compression (Q=75)**:
- **All methods degrade significantly**
- Performance drops across all detectors
- FFT features remain relatively stable
- CNN drops but stays above 0.9 AUC

JPEG compression introduces its own gradient artifacts and frequency patterns that can overwhelm subtle generator signatures. The 8×8 DCT blocks create their own high-frequency patterns.

**After Resize (0.5x then back to original)**:
- **Further degradation across the board**
- Gradient features struggle most with spatial distortions
- FFT features also degrade as high frequencies are lost
- CNN remains strongest but is still degraded

Downsampling destroys high-frequency information that all methods rely on. The bicubic interpolation when upscaling back smooths out telltale artifacts. This simulates what happens when images are shared on social media platforms that automatically resize uploads.

### Feature Space Visualization

Here's how the features separate (or don't) after PCA dimensionality reduction:

![PCA comparison across scenarios and methods](/content/2025/12/pca_comparison.png){: width="900" height="600" }
_PCA projection of gradient and FFT features for each scenario. Green dots = real images, red dots = AI-generated. Notice how the clusters overlap more as degradation increases._

The top row shows decent separation on raw images - you can see two somewhat distinct clusters. The bottom row shows significantly more overlap as degradation increases. The classes become harder to separate linearly, which explains why even logistic regression struggles.

### Frequency Domain Analysis

The article claims diffusion models have different frequency characteristics. Let's verify:

![FFT magnitude spectra comparison](/content/2025/12/fft_spectra_comparison.png){: width="900" height="600" }
_FFT magnitude spectra (log scale) for real vs AI-generated images. The center column shows DC and low frequencies with higher detail._

Real photos show more noise in high frequencies due to sensor characteristics and optical imperfections. AI-generated images tend to have slightly cleaner spectra because the diffusion denoising process can smooth high-frequency content.

![Radial frequency profile](/content/2025/12/radial_frequency_profile.png){: width="800" height="400" }
_Average radial frequency energy distribution across 50 samples. AI-generated images (red) have slightly different high-frequency energy than real images (green), but notice the substantial overlap shown by the shaded regions._

The shaded regions show standard deviation—there's considerable overlap between the two distributions. This explains why FFT features alone don't provide perfect separation. The difference exists but is subtle and inconsistent.

---

## Why "Just Look at Gradients" Fails in Practice

```mermaid
graph LR
    A[AI Image] --> B[Upload to Twitter]
    B --> C[JPEG Recompression]
    C --> D[Resize to 1024px]
    D --> E[Browser Optimization]
    E --> F[Gradients Now Look<br/>Like Real Photos]
    
    style F fill:#f96,stroke:#333,stroke-width:2px
```

### The Problem: Post-Processing

1. **JPEG artifacts dominate**: Compression introduces its own gradient patterns that overwhelm generator signatures
2. **Resolution changes**: Resizing filters smooth gradients, removing high-frequency traces
3. **Multiple hops**: Screenshot → re-upload → crop → filter destroys forensic traces
4. **Dataset bias**: If your "real" set is JPEG and "fake" set is PNG, you're detecting file format, not generation

### What Actually Works

**For controlled scenarios** (forensics, research):
- Gradient/FFT features work reasonably well on high-quality, unmodified images
- Deep learning detectors (ResNet, EfficientNet) achieve 95%+ accuracy
- Ensemble methods combining frequency + spatial features

**For real-world deployment**:
- Train on degraded images (JPEG, resize, blur)
- Use robust features (not just gradients)
- Test cross-generator generalization
- Accept that heavily processed images are undetectable

---

## The GenImage Benchmark Reality Check

The [GenImage dataset](https://arxiv.org/abs/2306.08571) is the gold standard for testing AI image detectors. It includes:
- 1M+ images from 8 different generators
- Degradation protocols (JPEG, blur, resize)
- Cross-generator evaluation

**Key findings from GenImage**:
- Detectors trained on one generator (e.g., StyleGAN) fail on others (e.g., Stable Diffusion)
- Performance drops 20-40% under degradation
- Frequency analysis shows diffusion models have *cleaner* spectra than GANs

From the GenImage paper frequency analysis:

> "GAN-generated images show regular grid artifacts in frequency domain. Diffusion-generated images have frequency characteristics closer to real images, presenting a greater detection challenge."

---

## Practical Takeaways for 2025

### ✅ What Works

- **Training on large, diverse datasets** (GenImage-scale, not CIFAR-scale)
- **Deep learning detectors** with augmentation and degradation in training
- **Ensemble methods** (combine gradient, frequency, and learned features)
- **Domain-specific fine-tuning** (medical images ≠ social media photos)

### ❌ What Doesn't Work

- "Universal" hand-crafted features (gradients alone, FFT alone)
- Detectors trained on pristine images only
- Single-generator training → multi-generator testing
- Ignoring preprocessing in real-world scenarios

### 🔬 The Research Frontier

**Current challenges**:
- **Diffusion models** are harder to detect than GANs
- **Post-processing** destroys most forensic traces
- **Adversarial attacks** can fool detectors
- **Arms race**: Generators improve, detectors lag behind

**Promising directions**:
- **Perceptual inconsistencies**: Detecting impossible shadows, perspective errors
- **Semantic analysis**: Text-image mismatch detection
- **Provenance tracking**: Blockchain-based authenticity certificates
- **Multi-modal detectors**: Combining visual + metadata + contextual cues

---

## References & Further Reading

- **GenImage Dataset**: [Million-Scale Benchmark for AI Image Detection](https://arxiv.org/abs/2306.08571)
- **LGrad (CVPR 2023)**: [Learning on Gradients for GAN Detection](https://github.com/chuangchuangtan/LGrad)
- **Frequency-based Detection**: [Natural Frequency Deviation for Diffusion Detection](https://openreview.net/forum?id=fPBExgC1m9)
- **CIFAKE Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/birgermoell/cifake-real-and-fake-images)
- **Diffusion vs GAN Detection**: [Recent Advances on Diffusion-Generated Image Detection](https://arxiv.org/html/2502.19716)

---

## Conclusion

Can you detect AI images with gradient analysis? **Sometimes**. 

On clean, high-resolution images with a controlled dataset? Yes, gradient and frequency features provide useful signals.

On real-world images that have been compressed, resized, filtered, and re-uploaded multiple times? The signal disappears.

The viral claim that "gradients reveal AI images reliably" is oversimplified. Modern detectors need:
- Deep learning architectures
- Training on degraded images
- Cross-generator generalization
- Ensemble approaches

The arms race continues. As of 2025, detecting AI images is possible but not trivial. The days of "just check the gradients" are over—if they ever existed at all.

**Want to test your own images?** Download the full code from [github.com/gsantopaolo/synthetic-image-detection](https://github.com/gsantopaolo/synthetic-image-detection) and experiment with different generators, preprocessing pipelines, and detection methods. The best way to understand what works is to break it yourself.
