"""Builds deepl_mid/midterm.ipynb (markdown + code cells). Temporary helper."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Image Classification on CIFAR-10: MLP vs CNN vs Transfer Learning (ResNet18)

**Deep Learning: Midterm Project**
Author: *FirstName LastName, Group X*

## Abstract
ABSTRACT_PLACEHOLDER
""")

md(r"""
## 1. Introduction

**Problem.** We classify small colour images into one of 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Input:** an RGB image of size 32×32×3.
- **Output:** one of 10 class labels.
- **Objective:** get the highest possible test accuracy, and understand *why* some models work better than others.

**Motivation.** Image classification is a core computer-vision task. It is a good fit for deep learning because the useful features (edges, textures, shapes) are hard to design by hand. Neural networks can learn them from data.

**Approach.**
1. **MLP baseline**: a fully connected network on flattened pixels.
2. **Custom CNN**: convolution + pooling layers that use the spatial structure of images.
3. **Optimizer comparison**: SGD, SGD+Momentum, RMSprop and Adam on the same CNN.
4. **Regularization experiment**: CNN with and without Dropout.
5. **Transfer learning**: ResNet18 pretrained on ImageNet, fine-tuned on CIFAR-10.
6. **Evaluation**: accuracy, loss curves, confusion matrices, parameter counts, training time, and filter/activation visualization.
""")

md("## 0. Setup (imports, seed, device)")
code(r"""
import time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Reproducibility: fix every random seed
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed()

# Use GPU (CUDA or Apple MPS) if available, otherwise CPU
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device, "| PyTorch:", torch.__version__)
""")

md(r"""
## 2. Data and Preprocessing

**Dataset:** CIFAR-10 (Krizhevsky, 2009). It is downloaded automatically with `torchvision.datasets.CIFAR10`.
- 60,000 colour images: 50,000 for training and 10,000 for testing.
- 10 classes, 32×32 pixels, 3 channels (RGB).

**Split (the same split is used for every model):**
- Train: 45,000 images (random 90% of the official training set, fixed seed)
- Validation: 5,000 images (the other 10%)
- Test: 10,000 images (official test set)

**Preprocessing:**
- `ToTensor()` scales pixels to [0, 1].
- `Normalize(mean, std)` uses the per-channel CIFAR-10 statistics, so each channel has about zero mean and unit variance. This makes optimization easier.
- **Augmentation (train only):** `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()`. These create slightly shifted or mirrored images, which reduces overfitting.
""")
code(r"""
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
plain_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])   # val / test (no augmentation)

root = "./data"
train_full_aug   = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
train_full_plain = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=plain_tf)
test_set         = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=plain_tf)
classes = train_full_aug.classes

# Fixed train/validation split (same indices for every experiment)
perm = torch.randperm(50000, generator=torch.Generator().manual_seed(SEED)).tolist()
train_idx, val_idx = perm[:45000], perm[45000:]
train_set       = Subset(train_full_aug,   train_idx)   # with augmentation
train_set_plain = Subset(train_full_plain, train_idx)   # without augmentation (dropout experiment)
val_set         = Subset(train_full_plain, val_idx)

BATCH = 128
train_loader       = DataLoader(train_set,       batch_size=BATCH, shuffle=True,
                                generator=torch.Generator().manual_seed(SEED))
train_loader_plain = DataLoader(train_set_plain, batch_size=BATCH, shuffle=True,
                                generator=torch.Generator().manual_seed(SEED))
val_loader  = DataLoader(val_set,  batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
print("Image shape (C, H, W):", tuple(train_set[0][0].shape), "| Classes:", len(classes))
""")

md("### Class distribution")
code(r"""
targets = np.array(train_full_aug.targets)
counts = pd.DataFrame({
    "train": np.bincount(targets[train_idx], minlength=10),
    "val":   np.bincount(targets[val_idx],   minlength=10),
    "test":  np.bincount(np.array(test_set.targets), minlength=10)}, index=classes)
display(counts.T)

counts["train"].plot.bar(figsize=(8, 3), title="Training-set class distribution", color="steelblue")
plt.ylabel("images"); plt.tight_layout(); plt.show()
""")
md("The classes are almost perfectly **balanced** (about 4,500 training images each), so accuracy is a fair metric.")

md("### Sample images")
code(r"""
raw = torchvision.datasets.CIFAR10(root, train=True, download=False)
fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for ax, i in zip(axes.flat, range(16)):
    img, y = raw[i]
    ax.imshow(img); ax.set_title(classes[y], fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Methods

### 3.1 Shared training setup
- **Loss function:** Cross-Entropy, the standard loss for multi-class classification. It combines softmax with negative log-likelihood.
- **Metric:** accuracy.
- The helper functions below train any model and record train/val loss and accuracy for every epoch, plus the training time.
""")
code(r"""
def count_params(model):
    # Number of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate(model, loader, criterion=nn.CrossEntropyLoss()):
    # Returns (average loss, accuracy) on a data loader
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n

def train_model(model, optimizer, epochs, loader=None, name=""):
    # Standard training loop. Returns a history dict.
    loader = loader or train_loader
    criterion = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()          # backpropagation
            optimizer.step()         # weight update
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        vl, va = evaluate(model, val_loader)
        hist["train_loss"].append(tot / n); hist["train_acc"].append(correct / n)
        hist["val_loss"].append(vl);        hist["val_acc"].append(va)
        print(f"[{name}] epoch {ep:2d}/{epochs} | train loss {tot/n:.3f} acc {correct/n:.3f} "
              f"| val loss {vl:.3f} acc {va:.3f}")
    hist["time"] = time.time() - start
    print(f"[{name}] training time: {hist['time']:.1f}s")
    return hist

def plot_history(hists, title=""):
    # Plot loss and accuracy curves for one or more runs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hists.items():
        ep = range(1, len(h["train_loss"]) + 1)
        ax[0].plot(ep, h["train_loss"], "-o", ms=3, label=f"{name} train")
        ax[0].plot(ep, h["val_loss"], "--", label=f"{name} val")
        ax[1].plot(ep, h["train_acc"], "-o", ms=3, label=f"{name} train")
        ax[1].plot(ep, h["val_acc"], "--", label=f"{name} val")
    ax[0].set(title=f"{title}: loss", xlabel="epoch", ylabel="loss")
    ax[1].set(title=f"{title}: accuracy", xlabel="epoch", ylabel="accuracy")
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()
""")

md(r"""
### 3.2 Baseline: Multilayer Perceptron (MLP)

| Layer | Details |
|---|---|
| Input | Flatten 3×32×32 → 3072 |
| Hidden 1 | Linear 3072 → 512, **ReLU**, Dropout 0.2 |
| Hidden 2 | Linear 512 → 256, **ReLU**, Dropout 0.2 |
| Output | Linear 256 → 10 (logits, softmax is inside the loss) |

- **Loss:** Cross-Entropy. **Optimizer:** Adam. **Learning rate:** 0.001. **Epochs:** 10. **Batch size:** 128.
- **Weakness:** an MLP treats every pixel independently. It ignores which pixels are neighbours and has no translation invariance, so we expect it to perform poorly on images.
""")
code(r"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

set_seed()
mlp = MLP().to(device)
print(mlp)
print("Trainable parameters:", f"{count_params(mlp):,}")

EPOCHS = 10
hist_mlp = train_model(mlp, torch.optim.Adam(mlp.parameters(), lr=1e-3), EPOCHS, name="MLP")
plot_history({"MLP": hist_mlp}, "MLP")
""")

md(r"""
### 3.3 Custom CNN

| Block | Layers | Output shape |
|---|---|---|
| Input | n/a | 3 × 32 × 32 |
| Conv block 1 | Conv2d(3→32, k=3, s=1, p=1) → ReLU → MaxPool(2) | 32 × 16 × 16 |
| Conv block 2 | Conv2d(32→64, k=3, s=1, p=1) → ReLU → MaxPool(2) | 64 × 8 × 8 |
| Conv block 3 | Conv2d(64→128, k=3, s=1, p=1) → ReLU → MaxPool(2) | 128 × 4 × 4 |
| Classifier | Flatten (2048) → Linear 2048→256 → ReLU → Dropout(p) → Linear 256→10 | 10 |

**Why these settings?**
- **Kernel size 3×3:** small and cheap. Stacking 3×3 kernels grows the receptive field (the VGG idea) with few parameters.
- **Number of filters 32 → 64 → 128:** the image gets smaller after each pooling, so we add more channels to keep enough capacity for more abstract features.
- **Stride 1:** the convolution looks at every position. Downsampling is done by pooling instead.
- **Padding 1:** with k=3 and p=1 the spatial size stays the same ("same" padding), so border information is not lost.
- **MaxPool 2×2:** halves height and width. This reduces computation and gives a little translation invariance.

**Output-size calculation.** Formula: $O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$

- Conv1: W=32, K=3, P=1, S=1 → (32 − 3 + 2)/1 + 1 = **32** → output 32×32×32
- MaxPool(2, stride 2): (32 − 2)/2 + 1 = **16** → 32×16×16
- Conv2: (16 − 3 + 2)/1 + 1 = 16 → pool → **64×8×8**
- Conv3: (8 − 3 + 2)/1 + 1 = 8 → pool → **128×4×4** → flatten = 2048

**Parameters of Conv1:** (3·3·3 + 1)·32 = 896. The code below checks the shapes and parameter counts.

**Training:** Cross-Entropy, Adam (lr = 0.001), 10 epochs, batch 128, Dropout p = 0.5.
""")
code(r"""
class CNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

# Check feature-map sizes layer by layer with a dummy image
x = torch.zeros(1, 3, 32, 32)
for layer in CNN().features:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape[1:])}")

# Parameters per layer
rows = [(n, p.numel()) for n, p in CNN().named_parameters()]
display(pd.DataFrame(rows, columns=["parameter", "count"]))
print("Total trainable parameters:", f"{count_params(CNN()):,}")
""")
code(r"""
set_seed()
cnn = CNN().to(device)
hist_cnn = train_model(cnn, torch.optim.Adam(cnn.parameters(), lr=1e-3), EPOCHS, name="CNN")
plot_history({"CNN": hist_cnn}, "CNN")
""")

md(r"""
### 3.4 Transfer learning: ResNet18 pretrained on ImageNet

**Idea.** A network trained on ImageNet (1.2M images, 1000 classes) has already learned general visual features: edges, colours, textures and object parts. We reuse these weights as the starting point and **fine-tune** them on our data. This usually gives higher accuracy with less training, because the model does not have to learn the low-level features from scratch.

**Steps:**
1. Load `resnet18` with `IMAGENET1K_V1` weights. ResNet uses *residual (skip) connections*, `y = F(x) + x`, which help gradients flow through deep networks.
2. Replace the final layer `fc: 512 → 1000` with `fc: 512 → 10` for our 10 classes.
3. Upsample inputs from 32×32 to 64×64. ResNet downsamples by 32×, so a 32×32 image would shrink to 1×1 too early.
4. Fine-tune **all** layers with Adam using a small learning rate (1e-4), so the pretrained features are not destroyed. We train for 3 epochs because the model converges fast.
""")
code(r"""
from torchvision.models import resnet18, ResNet18_Weights

def make_resnet():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # pretrained ImageNet weights
    m.fc = nn.Linear(m.fc.in_features, 10)                # new head for 10 classes
    return nn.Sequential(nn.Upsample(size=64, mode="bilinear", align_corners=False), m)

set_seed()
resnet = make_resnet().to(device)
print("Final layer:", resnet[1].fc)
print("Trainable parameters:", f"{count_params(resnet):,}")

hist_resnet = train_model(resnet, torch.optim.Adam(resnet.parameters(), lr=1e-4), 3, name="ResNet18")
plot_history({"ResNet18": hist_resnet}, "ResNet18 (fine-tuned)")
""")

md(r"""
## 4. Optimization and Training Analysis

### 4.1 Optimizer comparison
We train the **same CNN architecture** on the **same data split**, from the **same initial weights** (same seed), for 5 epochs with four optimizers:

| Optimizer | Learning rate | Idea |
|---|---|---|
| SGD | 0.01 | plain gradient step: w ← w − η∇L |
| SGD + Momentum | 0.01, momentum 0.9 | adds a "velocity" that smooths updates and speeds up movement in consistent directions |
| RMSprop | 0.001 | divides by a running average of squared gradients, so each parameter gets its own adaptive step |
| Adam | 0.001 | momentum + RMSprop-style adaptive steps, with bias correction |
""")
code(r"""
OPT_EPOCHS = 5
optimizers = {
    "SGD":          lambda p: torch.optim.SGD(p, lr=0.01),
    "SGD+Momentum": lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    "RMSprop":      lambda p: torch.optim.RMSprop(p, lr=0.001),
    "Adam":         lambda p: torch.optim.Adam(p, lr=0.001),
}
opt_hists = {}
for name, make_opt in optimizers.items():
    set_seed()                       # identical initial weights for a fair comparison
    model = CNN().to(device)
    opt_hists[name] = train_model(model, make_opt(model.parameters()), OPT_EPOCHS, name=name)
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, h in opt_hists.items():
    ep = range(1, OPT_EPOCHS + 1)
    ax[0].plot(ep, h["train_loss"], "-o", label=name)
    ax[1].plot(ep, h["val_acc"], "-o", label=name)
ax[0].set(title="Training loss per optimizer", xlabel="epoch", ylabel="loss")
ax[1].set(title="Validation accuracy per optimizer", xlabel="epoch", ylabel="accuracy")
for a in ax: a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.show()

opt_table = pd.DataFrame({
    "final train loss": [h["train_loss"][-1] for h in opt_hists.values()],
    "final val acc":    [h["val_acc"][-1]    for h in opt_hists.values()],
    "best val acc":     [max(h["val_acc"])   for h in opt_hists.values()],
    "time (s)":         [h["time"]           for h in opt_hists.values()],
}, index=opt_hists.keys()).round(3)
opt_table
""")
md("OPT_DISCUSSION_PLACEHOLDER")

md(r"""
### 4.2 Regularization experiment: Dropout

**Dropout** randomly sets a fraction *p* of neurons to zero during training. The network cannot rely on any single neuron, so it learns more robust and redundant features. At test time all neurons are used (PyTorch rescales automatically). Dropout acts like averaging many smaller networks, which reduces **overfitting**.

**Setup:** the same CNN with `dropout=0.0` vs `dropout=0.5`, Adam lr=0.001, 10 epochs. We train **without data augmentation** here so that overfitting is easy to see. We compare the **gap between training and validation** performance.
""")
code(r"""
reg_hists = {}
for p in [0.0, 0.5]:
    set_seed()
    model = CNN(dropout=p).to(device)
    reg_hists[f"dropout={p}"] = train_model(model, torch.optim.Adam(model.parameters(), lr=1e-3),
                                            EPOCHS, loader=train_loader_plain, name=f"dropout={p}")
plot_history(reg_hists, "Dropout experiment (no augmentation)")

pd.DataFrame({
    "final train acc": [h["train_acc"][-1] for h in reg_hists.values()],
    "final val acc":   [h["val_acc"][-1]   for h in reg_hists.values()],
    "train-val gap":   [h["train_acc"][-1] - h["val_acc"][-1] for h in reg_hists.values()],
    "min val loss":    [min(h["val_loss"]) for h in reg_hists.values()],
    "final val loss":  [h["val_loss"][-1]  for h in reg_hists.values()],
}, index=reg_hists.keys()).round(3)
""")
md("REG_DISCUSSION_PLACEHOLDER")

md(r"""
## 5. Experiments and Results

### 5.1 Quantitative comparison (test set)
""")
code(r"""
models = {"MLP": (mlp, hist_mlp), "CNN": (cnn, hist_cnn), "ResNet18 (TL)": (resnet, hist_resnet)}
results = []
for name, (m, h) in models.items():
    _, test_acc = evaluate(m, test_loader)
    results.append({"model": name, "test accuracy": round(test_acc, 4),
                    "trainable params": f"{count_params(m):,}",
                    "epochs": len(h["train_loss"]),
                    "training time (s)": round(h["time"], 1),
                    "time / epoch (s)": round(h["time"] / len(h["train_loss"]), 1)})
results = pd.DataFrame(results).set_index("model")
results
""")

md("### 5.2 Training and validation loss curves (all models)")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, (_, h)) in zip(axes, models.items()):
    ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], "-o", label="train")
    ax.plot(ep, h["val_loss"], "-s", label="validation")
    ax.set(title=name, xlabel="epoch"); ax.legend(); ax.grid(alpha=.3)
axes[0].set_ylabel("cross-entropy loss")
plt.tight_layout(); plt.show()
""")

md("### 5.3 Confusion matrices (test set)")
code(r"""
@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, (m, _)) in zip(axes, models.items()):
    y_true, y_pred = predict(m, test_loader)
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{name}  (acc = {(y_true == y_pred).mean():.3f})")
plt.tight_layout(); plt.show()
""")
md("CM_DISCUSSION_PLACEHOLDER")

md(r"""
### 5.4 Feature visualization

**(a) Filters of the first convolutional layer.** Conv1 has 32 filters of size 3×3×3. We show them as tiny RGB images.
""")
code(r"""
w = cnn.features[0].weight.detach().cpu()          # shape (32, 3, 3, 3)
w = (w - w.min()) / (w.max() - w.min())             # scale to [0, 1] for display
fig, axes = plt.subplots(2, 16, figsize=(14, 2.2))
for i, ax in enumerate(axes.flat):
    ax.imshow(w[i].permute(1, 2, 0)); ax.axis("off"); ax.set_title(str(i), fontsize=7)
plt.suptitle("All 32 learned filters of Conv1 (3×3 RGB)"); plt.tight_layout(); plt.show()
""")
md("**(b) Activation maps of Conv1 for two test images** (first 8 channels after ReLU).")
code(r"""
cnn.eval()
fig, axes = plt.subplots(2, 9, figsize=(16, 4))
for row, idx in enumerate([3, 7]):                  # two sample test images
    img, y = test_set[idx]
    with torch.no_grad():
        act = cnn.features[1](cnn.features[0](img.unsqueeze(0).to(device)))[0].cpu()  # Conv1 + ReLU
    show = (img * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
    axes[row, 0].imshow(show.permute(1, 2, 0)); axes[row, 0].set_title(f"input: {classes[y]}")
    for c in range(8):
        axes[row, c + 1].imshow(act[c], cmap="viridis"); axes[row, c + 1].set_title(f"map {c}", fontsize=8)
for ax in axes.flat: ax.axis("off")
plt.tight_layout(); plt.show()
""")
md(r"""
**Interpretation.** The first-layer filters look like **colour blobs and oriented edge detectors** (light/dark transitions in different directions, plus some colour-opponent filters). In the activation maps, some channels light up on the **object outline/edges**, others on **uniform colour regions** (sky, background), and others on specific colours. So early CNN layers capture **low-level features**: edges, colours and simple textures. Deeper layers combine these into shapes and object parts.
""")

md("### 5.5 Discussion: complexity vs training cost vs performance\nTRADEOFF_PLACEHOLDER")

md("## 6. Conclusion\nCONCLUSION_PLACEHOLDER")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
