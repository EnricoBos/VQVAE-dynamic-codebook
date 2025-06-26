# VQ-VAE for Face Image Reconstruction

This repository implements a **Vector Quantized Variational Autoencoder (VQVAE)** for reconstructing facial images using the **LFW deepfunneled** dataset. The model includes training features such as Exponential Moving Average (EMA) updates, dynamic commitment cost annealing, and dead code revival, all designed to enhance codebook utilization and reconstruction quality.

---

## Overview

The project focuses on:
- Good-quality reconstruction of facial images from the **LFW deepfunneled** dataset using **VQVAE**
- Comparing reconstruction quality against standard VAEs
- A training pipeline with:
  - ✅ **Exponential Moving Average (EMA)** updates for the codebook 
  - ✅ **Adaptive commitment cost annealing** based on code usage
  - ✅ **Dead code revival** to prevent embedding collapse
  - ✅ **Image augmentation** for better generalization

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

## Data Processing

### Image Preprocessing
- All images are cropped and resized to a consistent resolution suitable for model input(and computational power avaibale).
- Pixel values are normalized to the range \([0, 1]\) by dividing by 255.

### Data Augmentation
To improve generalization and robustness, multiple augmentations steps preserving attrivute are applied to each image using `PIL`:
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

Each augmented image preserves the core attributes of the original!.

---

## Dependencies

Install the required packages:

```bash
conda env create -f environment.yaml
```

Main dependencies:
- `tensorflow`
- `numpy`, `pandas`
- `Pillow`, `imageio`
- `scikit-learn`

---

## Usage

### Training

Set the mode in `vqvae_face_main.py`:

```python
def main():
    
    ### ENABLING TRAINING OR LOADING MODEL PERFORMING RECOSTRUCTION CHECK
    enable_trainig = True
```

- Loads and augments the dataset
- Adjusts commitment cost based on active code usage
- Applies EMA updates to the codebook
- Saves weights and checkpoints in `checkpoints/`

### Evaluation

Set:

```python
def main():
    ### ENABLING TRAINING OR LOADING MODEL PERFORMING RECOSTRUCTION CHECK
    enable_trainig = False
```

- Loads the trained model and tokenized images
- Reconstructs and saves sample outputs for comparison

---

## Quantization Strategy

The quantization pipeline in this project includes two major mechanisms:

### EMA-Based Codebook Updates
The codebook embeddings are updated using an **Exponential Moving Average (EMA)** strategy rather than gradients. This makes training more stable and prevents code collapse by gradually integrating the encoder's outputs into the codebook.

### Dead Code Revival
To prevent embedding collapse and ensure full codebook utilization, dead embeddings are periodically identified and revived.

- A code is considered "dead" if its usage falls below a defined threshold.
- Up to 5 dead codes per batch are replaced using noisy copies of active embeddings, allowing them to re-enter training gradually.
- A fallback is included to reinitialize the entire codebook if all embeddings go inactive.
- This strategy maintains a healthy distribution of latent representations and prevents the model from underutilizing its capacity.

---

## Example Reconstructions

| Original VS Reconstructed (VQ-VAE) |

<img width="720" alt="original_vs_reconstructions_epoch" src="https://github.com/user-attachments/assets/d952c2d6-5042-46f0-a743-22e1b1d533e2" />

---

## Reference
This implementation was inspired in part by the official Keras VQ-VAE example:
 keras.io/examples/generative/vq_vae

## Author

**Enrico Boscolo**  
