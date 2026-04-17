# HW4 - Generative Adversarial Networks (GAN)
 Grade: 100  

---

##  Overview
In this project, I implemented a **Generative Adversarial Network (GAN)** to generate realistic images from random noise.

The model was trained on the Oxford 102 Flower Dataset and learned to synthesize new, diverse flower images that resemble real samples from the dataset.

This project focuses on **unsupervised learning**, where the model learns the data distribution without explicit labels.

---

## Key Concepts
- Generative Adversarial Networks (GAN)
- Generator vs Discriminator training
- Latent space representation
- Image synthesis from noise
- Training stability in GANs

---

## Model Architecture

The GAN consists of two competing networks:

### 🔹 Generator
- Input: Random latent vector \( z \sim \mathcal{N}(0,1) \)
- Output: Synthetic image
- Uses transposed convolutions (`ConvTranspose2d`) to upscale noise into images

### 🔹 Discriminator
- Input: Image (real or generated)
- Output: Probability that the image is real
- Learns to distinguish between real and fake images

 The two networks are trained in an adversarial manner:
- Generator tries to fool the discriminator
- Discriminator tries to correctly classify real vs fake images

---

##  Training Process

- Dataset: Oxford 102 Flower Dataset
- Images resized to maintain quality (≥ 64x64)
- Alternating training between generator and discriminator

### Loss Functions:
- Generator Loss – encourages realistic image generation
- Discriminator Loss – improves real/fake classification

### Training Challenges:
- Mode collapse
- Training instability
- Sensitivity to hyperparameters

These were addressed באמצעות:
- Careful tuning of learning rates
- Balanced training between networks

---

##  Image Generation

After training, the generator can produce new images:

- Sample latent vector \( z \)
- Pass through generator
- Output synthetic flower image

### Results:
- Generated realistic and diverse flower images
- Model learned meaningful visual patterns
- No memorization of training data

---

##  Latent Space Analysis

- Explored relationships between generated samples
- Compared similar vs dissimilar images using L2 distance
- Observed that:
  - Similar images are closer in latent space
  - Latent vectors capture semantic structure

---

##  Files Description

### 🔹 `hw4_code.py`
Main implementation file.

Contains:
- Generator and Discriminator architecture
- Training loop
- Loss computation
- Optimization process

---

### 🔹 `hw4_generation.py`
Inference and generation script.

Contains:
- `reproduce_hw4` function
- Loads trained model
- Generates new images from random latent vectors

---

### 🔹 `hw4_report.pdf`
Project report.

Includes:
- Model design explanation
- Training process
- Results analysis
- Discussion of challenges (e.g., mode collapse)
- Visualization of generated images

---
### 🔹 `model_weights.pkl`
Trained model weights.

- Saved after training
- Used for generating images without retraining
- Due to file size limitations, this file is not included in the repository
- The weights can be provided separately (e.g., via Google Drive) if needed

### 🔹 `requirements.txt`
Dependencies used in the project.

- Enables reproducibility

