# Music Genre Classification — CNN Built From Scratch

A 16-class music genre classifier trained on three complementary audio representations
(mel spectrogram, CQT, chromagram). Built for the CS776 (Deep Learning for Computer Vision) course
contest at IIT Kanpur, where every layer had to be implemented manually in raw PyTorch tensor
operations.

---

## The constraint

The contest banned essentially all of PyTorch's neural network toolkit. Not allowed:

- `nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`, `nn.MaxPool2d`, `nn.Dropout`, `nn.ReLU`, `nn.Sequential`
- `F.conv2d`, or any wrapper calling optimized convolution kernels
- Any pretrained weights, transfer learning, or imported architecture (ResNet/VGG/timm/torchvision)
- Any external dataset

Allowed: raw tensors, autograd, optimizers, `F.unfold`/`F.pad`, and `nn.Module` only as a base class
for parameter registration. Hard cap of 500K parameters, and everything had to train end-to-end on
Kaggle's free GPU.

## Architecture

```
Input: 3 × 128 × 128   (mel spectrogram | CQT | chromagram, stacked as channels)

Block 1:  3 →  28   5×5 conv, pad 2  → BN → LeakyReLU → MaxPool 2×2 → SpatialDropout 0.04
Block 2: 28 →  52   3×3 conv, pad 1  → BN → LeakyReLU → MaxPool 2×2 → SpatialDropout 0.05
Block 3: 52 →  96   3×3 conv, pad 1  → BN → LeakyReLU → MaxPool 2×2 → SpatialDropout 0.06
Block 4: 96 → 144   3×3 conv, pad 1  → BN → LeakyReLU → MaxPool 2×2 → SpatialDropout 0.07
Block 5: 144 → 200  3×3 conv, pad 1  → BN → LeakyReLU →      —       → SpatialDropout 0.08

Global Average Pooling  →  200-d vector
FC 200 → 128  → LeakyReLU → Dropout 0.5
FC 128 →  16  → logits
```

Design decisions:

- **GAP instead of flatten.** Flattening 200×8×8 into a fully connected layer would consume the
  entire 500K budget on its own. Global average pooling collapses it to 200 dimensions.
- **5×5 first kernel, 3×3 after.** Spectrogram patterns are broad at the input, so the first layer
  gets a wider receptive field.
- **BN before activation.** Normalizing the pre-activation distribution is what allowed the 0.0035
  learning rate that OneCycle needs.
- Block 5 skips max pooling. Feature maps are already down to 8×8 by that point.
- Spatial dropout increases across blocks, from 0.04 at block 1 to 0.08 at block 5.

Parameter budget: 473,100 of 500,000, or 94.6% of the cap.

| Component | Params |
|---|---:|
| Conv1 (3→28, 5×5) | 2,128 |
| Conv2 (28→52, 3×3) | 13,156 |
| Conv3 (52→96, 3×3) | 45,024 |
| Conv4 (96→144, 3×3) | 124,560 |
| Conv5 (144→200, 3×3) | 259,400 |
| BatchNorm (all 5 blocks) | 1,040 |
| FC1 (200→128) | 25,728 |
| FC2 (128→16) | 2,064 |
| **Total** | **473,100** |

## Data pipeline

Three grayscale representations of the same audio clip → resized to 128×128 → concatenated into a
3-channel tensor → normalized to [-1, 1].

Augmentation differs by representation:

- Mel and CQT get SpecAugment (time and frequency masking) plus a random circular time shift of
  ±15 frames.
- **Chromagram gets time masking only.** Frequency masking would destroy the 12-bin harmonic
  structure, since each chroma bin is a pitch class rather than an arbitrary frequency band.

Class distribution ranges from 3.5% to 10.9% per class. This is handled with inverse-frequency class
weights in the cross-entropy loss rather than resampling.

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| Scheduler | OneCycleLR, max LR 0.0035, 10% warmup |
| Loss | Weighted cross-entropy, label smoothing 0.1 |
| Batch size | 64 |
| Epochs | 90 |
| Gradient clipping | max norm 1.5 |
| Split | 80/20 random with a fixed seed, 17,460 train / 4,365 validation |
| Selection | Best checkpoint by validation macro F1 |

OneCycleLR with AdamW reached good performance around epoch 40–50, where step-based schedules needed
70–80.

## Results

Contest leaderboard macro F1: **0.93924**

Held-out validation split (4,365 samples):

| Metric | Score |
|---|---|
| Accuracy | 94.16% |
| Precision (macro) | 0.9369 |
| Recall (macro) | 0.9393 |
| **F1 (macro)** | **0.9378** |


## Repository

```
music-genre-classification/
├── music_genre_classification.ipynb   # complete pipeline, runs top to bottom
└── README.md
```

The notebook is self-contained and executes sequentially without manual intervention: manual layer
implementations → model → augmentation → training → evaluation (accuracy, macro precision/recall/F1,
confusion matrix, per-class metrics, parameter count) → test inference.

## Context

CS776 Deep Learning for Computer Vision course contest, IIT Kanpur.

My contribution: initial and final CNN architecture design, manual layer implementations, weight
initialization, preprocessing pipeline, class imbalance handling, hyperparameter tuning.
