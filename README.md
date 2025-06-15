# VQ-VAE for High-Fidelity Face Image Reconstruction

This repository implements a **Vector Quantized Variational Autoencoder (VQ-VAE)** for reconstructing low-resolution facial images using the **LFW deepfunneled** dataset. The model includes advanced training features such as Exponential Moving Average (EMA) updates, dynamic commitment cost annealing, and dead code revival, all designed to enhance codebook utilization and reconstruction quality.

---

## 🧠 Overview

The project focuses on:
- High-quality reconstruction of facial images from the **LFW deepfunneled** dataset using **VQ-VAE**
- Comparing reconstruction quality against standard VAEs on low-resolution inputs
- A robust training pipeline featuring:
  - ✅ **Exponential Moving Average (EMA)** updates for the codebook (VQ-VAE v2 style)
  - ✅ **Adaptive commitment cost annealing** based on code usage
  - ✅ **Dead code revival** to prevent embedding collapse
  - ✅ Extensive **image augmentation** for regularization and evaluation robustness

---

## 📁 Project Structure

```
* vqvae_face_main.py     : Main script for training and evaluating the VQ-VAE model  
* vqvae_class_keras.py   : Contains the VQ-VAE model implementation (encoder, decoder, quantizer)  
* experiment.json        : JSON file containing model configuration parameters and training settings  
* checkpoints/           : Directory for saving model checkpoints during training  
* images_saved/          : Directory for saving images generated during training and evaluation  
```

---

## 🧪 Data Processing

### 🖼️ Image Preprocessing
- All images are cropped and resized to a consistent resolution suitable for model input.
- Pixel values are normalized to the range \([0, 1]\) by dividing by 255.

### 🔁 Data Augmentation
To improve generalization and robustness, multiple **attribute-preserving augmentations** are applied to each image using `PIL`:
- **Horizontal Flip**
- **Brightness** adjustment (±20%)
- **Contrast** variation (±30%)
- **Sharpness** variation (±50%)
- **Saturation (Color)** adjustment (±20%)
- **Rotation** (±5 degrees)
- **Gaussian Noise** injection
- **Slight Translation** (random pixel shifts)
- **Gaussian Blur**
- **Zoom-in** cropping (90% of the original image)

Each augmented image preserves the core attributes of the original, enabling the model to learn invariant features while increasing data diversity.

---

## 📦 Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `tensorflow`
- `numpy`, `pandas`
- `Pillow`, `imageio`
- `scikit-learn`

---

## 🚀 Usage

### Training

Set the mode in `vqvae_face_main.py`:

```python
if __name__ == "__main__":
    choice = 'train'
```

- Loads and augments the dataset
- Adjusts commitment cost based on active code usage
- Applies EMA updates to the codebook
- Saves weights and checkpoints in `checkpoints/`

### Evaluation

Set:

```python
if __name__ == "__main__":
    choice = 'eval'
```

- Loads the trained model and tokenized images
- Reconstructs and saves sample outputs for comparison

---

## 🧠 Quantization Strategy

The quantization pipeline in this project includes two major mechanisms:

### 🔁 EMA-Based Codebook Updates
The codebook embeddings are updated using an **Exponential Moving Average (EMA)** strategy rather than gradients. This makes training more stable and prevents code collapse by gradually integrating the encoder's outputs into the codebook.

### 🧪 Dead Code Revival
Unused (dead) embeddings are periodically identified and reinitialized using slightly perturbed versions of active embeddings. This keeps the codebook fully utilized and avoids representational bottlenecks in the latent space.

---

## 🖼 Example Reconstructions

| Original | Reconstructed (VQ-VAE) |
|----------|------------------------|
| ![o1](images_saved/processed_original_1.png) | ![r1](images_saved/recon_1.png) |

---

## 👨‍💻 Author

**Enrico Boscolo**  
For questions, contributions, or collaboration, feel free to open an issue or contact me directly.