# GAN on MNIST — Generative Adversarial Network with TensorFlow

A from-scratch implementation of a **Deep Convolutional GAN (DCGAN)** trained on the MNIST handwritten digits dataset using TensorFlow / Keras.

---

## Overview

This notebook walks through building and training a GAN that learns to generate realistic handwritten digit images (28×28 px, grayscale) from random noise vectors.

The two adversarial networks are:

- **Generator** — takes a 100-dimensional noise vector and upsamples it into a 28×28 image via transposed convolutions.
- **Discriminator** — a convolutional classifier that distinguishes real MNIST images from generated fakes.

Both networks are trained simultaneously in a minimax game until the generator fools the discriminator.

---

## Architecture

### Generator
| Layer | Details |
|---|---|
| Dense | 7×7×256, no bias |
| BatchNormalization + LeakyReLU | — |
| Reshape | (7, 7, 256) |
| Conv2DTranspose | 128 filters, 5×5, stride 1 |
| BatchNormalization + LeakyReLU | — |
| Conv2DTranspose | 64 filters, 5×5, stride 2 → 14×14 |
| BatchNormalization + LeakyReLU | — |
| Conv2DTranspose | 1 filter, 5×5, stride 2 → 28×28, **tanh** |

### Discriminator
| Layer | Details |
|---|---|
| Conv2D | 64 filters, 5×5, stride 2 |
| LeakyReLU + Dropout(0.3) | — |
| Conv2D | 128 filters, 5×5, stride 2 |
| LeakyReLU + Dropout(0.3) | — |
| Flatten + Dense(1) | Logit output |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 256 |
| Noise dimension | 100 |
| Optimizer | Adam (lr = 1e-4) |
| Loss | Binary Cross-Entropy (from logits) |
| Image normalization | [−1, 1] |

A preview grid of 16 generated images is displayed every 10 epochs, and a final visualization is shown after training.

---

## Requirements

```
tensorflow >= 2.x
numpy
matplotlib
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## Usage

Open and run the notebook cell by cell:

```bash
jupyter notebook gan-mnist.ipynb
```

Or run it end-to-end in [Google Colab](https://colab.research.google.com/) (GPU recommended).

---

## Results

After 50 epochs of training, the generator produces images that resemble handwritten digits from the MNIST dataset. Longer training (100+ epochs) generally improves sharpness and diversity.

---

## References

- [Goodfellow et al., 2014 — Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- [TensorFlow DCGAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
