# Galaxy Morphology Classification & Representation Learning

This repository contains experiments using deep learning for galaxy image analysis.  

---

## Dataset

- ~8,000 RGB galaxy images  
- Resolution 64×64×3  
- Classes include elliptical, barred/unbarred spiral, tight/loose spirals, etc.  
- High visual similarity across subclasses makes classification challenging  

---

## 1) CNN Classifier

### Architecture Overview

- Kernel size kept small (3×3) to avoid oversmoothing on small 64×64 inputs  
- Progressive filter schedule: `32 → 64 → 128`  
- Block structure used: `Conv2D → BatchNorm → ReLU → MaxPool(2×2)`  

- Avoided `Flatten()` because dense head becomes too large  
  (Flattening → ~1M parameters → heavy overfitting risk)

### Regularization Techniques Applied

| Method | Effect |
|---|---|
| Input noise | Reduced validation loss significantly |
| Dropout in dense layers | Prevented memorization |
| L2 regularization | Helped generalize better |
| Label smoothing | Improved soft decision boundary |
| Batch size increase (32→64) | Higher validation accuracy |

### Performance

- Test Accuracy: **~0.82**
- Frequent confusion between class 5 (Barred Spiral) and 6 (Unbarred Tight Spiral)

---

## 2) Variational Autoencoder + Normalizing Flows

Goal: Improve latent representation quality for realistic galaxy generation.  
Normalizing flows were added to map a simple Gaussian prior into a more flexible latent distribution.

### Important Training Notes

- Added **β-VAE KL weighting**:  
  `beta * KL_loss` helps avoid the latent collapse → model captures colour gradients better  
  This reduces the tendency for decoded outputs to drift toward grey/0.5 uniform values.  
  The encoder no longer falls into the "neutral grey" minimum → noticeably richer colour reconstruction.

### Sampling Code

```python
z0 = tf.random.normal((num_samples, LATENT_DIM))  # Base Gaussian
zK, _ = flow(z0, training=False)                  # Flow transforms latent
images = vae.decoder(zK)                          # Decode to galaxy images
```
#### Step 1: Sample `z0` from a standard Gaussian
The VAE assumes the latent prior is `N(0, I)`. So generation starts by drawing noise from this base distribution:

```python
z0 = tf.random.normal(...)
```
This is NOT yet the latent the model expects, it's just the unwarped space.

#### Step 2: Push that Gaussian sample through the Normalizing Flow:
A normalizing flow learns an invertible transformation that reshapes simple Gaussian noise into the true data latent distribution.

```python
zK, logdet = flow(z0, training=False)
```

Meaning:

```bash
z0  →  z1  →  z2  →  ...  →  zK  = learned latent space
```

- The VAE encoder learns one approximation of the latent space,
- The flow further warps and densifies it into a more realistic manifold.

**Results**:
Sharper details, improved colour representation, and more diverse morphologies.

### Why training=False is important
Because at inference we want a *deterministic* mapping.

| If `training=True` | If `training=False` |
|---|---|
| Flow still tries to learn | Flow **uses** what it learned |
| Latent distribution **shifts** | Latent stays correct |
| Images look worse / unstable | Images are clean + realistic |

So for sampling **we freeze everything** → only use what was learned.

---

## 3) Convolutional Autoencoder (Anomaly Detection)

Used to reconstruct galaxies and detect anomalies via reconstruction error.

### Preprocessing
```python
images = np.arcsinh(images / 20.0)
channel_max = images.reshape(-1, 3).max(axis=0)
images = images / channel_max
```
#### 1) `np.arcsinh(images / 20.0)`
- This is a **non-linear** compression, similar to logarithmic stretching.
- Astronomers use `arcsinh` because it:
  - Preserves faint structures (outer spiral arms, halos)
  - Prevents bright cores from blowing out white
- Empirically, `arcsinh` gave sharper features and improved ROI

#### 2) `channel_max = images.reshape(-1, 3).max(axis=0)`
- The image tensor is flattened → shape (N, 3)
- We then compute the maximum intensity per channel (R/G/B separately)
- Result example: `channel_max = [max_R, max_G, max_B]`

#### 3) `images = images / channel_max`
- Each image is normalized by its channel-wise maximum
- Ensures all channels are scaled to `[0,1]` range
- Prevents any color channel from dominating

### Current Limitation

Even with tuning, anomaly ROC ≈ **0.51**, meaning:

- Model struggles to separate normal vs. anomalous galaxies  
- AE may be reconstructing anomalies too cleanly  
- Low morphological contrast = hard boundary to detect

---

### Next Step — Add AHUNT Pipeline

Found a method that directly targets galaxy anomaly detection:

*[Personalized anomaly detection using deep active learning](https://ui.adsabs.harvard.edu/abs/2023RASTI...2..586V/abstract)*  (2023) — RASTI Journal  
Could be integrated to significantly improve anomaly separation.

> Recommended expansion: implement AHUNT downstream of VAE embeddings for more robust anomaly detection.


