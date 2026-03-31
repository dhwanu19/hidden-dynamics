# Hidden Dynamics in Neural Networks

A deep-dive research and education project exploring the **internal representations and hidden-state dynamics** of recurrent neural networks (RNNs), Long Short-Term Memory (LSTM) networks, multi-layer perceptrons (MLPs), and convolutional neural networks (CNNs). The project visualises how learned hidden states evolve over time and across tasks, providing geometric and analytical insight into what these networks actually learn.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Background & Motivation](#background--motivation)
- [Repository Structure](#repository-structure)
- [Experimental Modules](#experimental-modules)
  - [1. Formal Language Learning (RNN / LSTM)](#1-formal-language-learning-rnn--lstm)
  - [2. Binary Classification (MLP)](#2-binary-classification-mlp)
  - [3. Image Classification (KMNIST / CNN)](#3-image-classification-kmnist--cnn)
- [Model Architectures](#model-architectures)
  - [Simple Recurrent Network (SRN)](#simple-recurrent-network-srn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
  - [Image Classification Networks](#image-classification-networks)
- [Datasets](#datasets)
- [Dependencies & Installation](#dependencies--installation)
- [Usage](#usage)
  - [Training Sequence Models (SRN / LSTM)](#training-sequence-models-srn--lstm)
  - [Visualising Sequence Model Trajectories](#visualising-sequence-model-trajectories)
  - [Training the MLP Binary Classifier](#training-the-mlp-binary-classifier)
  - [Training the KMNIST Image Classifier](#training-the-kmnist-image-classifier)
  - [Plotting Hidden-State Trajectories](#plotting-hidden-state-trajectories)
  - [Parsing Training Output](#parsing-training-output)
- [Key Results & Visualisations](#key-results--visualisations)
- [Technical Deep Dive](#technical-deep-dive)
  - [Hidden State Dynamics](#hidden-state-dynamics)
  - [LSTM Context (Cell) Vectors](#lstm-context-cell-vectors)
  - [Backpropagation Through Time (BPTT)](#backpropagation-through-time-bptt)
- [File-by-File Reference](#file-by-file-reference)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Extending the Project](#extending-the-project)
- [Academic Context](#academic-context)

---

## Project Overview

This project studies **what neural networks internally represent** by:

1. Training RNN / LSTM models on **synthetic formal languages** (`aⁿb²ⁿ`, `aⁿb²ⁿc³ⁿ`) and inspecting the geometry of the learned hidden-state space.
2. Training an **MLP** on a 2-D binary classification task and visualising the hidden-unit activations and decision boundary.
3. Training **CNN / fully-connected** networks on the **KMNIST** handwritten-kanji benchmark and evaluating them with per-class confusion matrices.
4. Producing **2-D and 3-D trajectory plots** of hidden states and LSTM cell (context) vectors to reveal how a network tracks the counting variable embedded in a formal language.

The central question driving the project is:

> *How does a recurrent network encode an unbounded integer counter needed to accept `aⁿb²ⁿ` while only using a finite-dimensional real-valued state space?*

---

## Background & Motivation

Recurrent neural networks operate by maintaining a hidden state `h_t` that is updated at every time step. For a network to correctly accept sequences of the form `aⁿb²ⁿ` it must effectively *count* how many `a` characters it has seen and then produce exactly twice as many `b` characters. This requires the network to store an unbounded integer — a feat that is non-trivial for a fixed-dimensional continuous state vector.

Prior theoretical work shows that an RNN with as few as two hidden units can, in principle, represent integer counting with a spiral trajectory in 2-D state space. This project implements and visualises that phenomenon empirically, extending the analysis to LSTM networks whose **cell state** (context vector) provides an alternative channel for information storage.

Key concepts covered:

| Concept | Relevance |
|---|---|
| Recurrent Neural Networks | Core sequential model |
| LSTM gates (input, forget, output, cell) | Extended memory mechanism |
| Backpropagation Through Time (BPTT) | Training RNNs on variable-length sequences |
| Hidden-state trajectory geometry | Interpretability / mechanistic understanding |
| Formal language theory (regular vs. context-free) | Task design — `aⁿb²ⁿ` is context-free |
| Convolutional Neural Networks | Spatial feature extraction for images |
| Transfer learning / representation analysis | Visualising what layers learn |

---

## Repository Structure

```
hidden-dynamics/
│
├── seq_models.py          # SRN and LSTM PyTorch model definitions
├── seq_train.py           # CLI training script for sequence models
├── seq_plot.py            # Loads trained models and produces hidden-state plots
│
├── anb2n.py               # Synthetic formal-language generator (aⁿb²ⁿ / aⁿb²ⁿc³ⁿ)
│
├── check.py               # MLP (binary classifier) definition
├── check_main.py          # Training + visualisation script for MLP
│
├── kuzu.py                # NetLin / NetFull / NetConv definitions for KMNIST
├── kuzu_main.py           # Training + evaluation (confusion matrix) for KMNIST
│
├── plot_traj.py           # Multi-subplot trajectory plotter (4 × sub-figures per sequence)
├── hidden_traj_all.py     # 3-D trajectory plot — hidden states across all test sequences
├── context_traj_all.py    # 3-D trajectory plot — LSTM context (cell) vectors
│
├── text_to_dict.py        # Parses raw training-log text into a Python dict
│
├── a1.pdf                 # Assignment specification (41 pages)
└── README.md              # This file
```

---

## Experimental Modules

### 1. Formal Language Learning (RNN / LSTM)

**Formal languages under study:**

| Language | Alphabet | Rule | Type |
|---|---|---|---|
| `anb2n` | {A, B} | n A's followed by 2n B's | Context-free |
| `anb2nc3n` | {A, B, C} | n A's, 2n B's, 3n C's | Context-sensitive |

**How training works:**

- Sequences are generated stochastically by `anb2n.py` on-the-fly; no static dataset file is required.
- The network receives one-hot encoded characters as inputs one step at a time.
- The target at each step is a **probability distribution** over the next character (not a hard label), reflecting the fact that when the network has seen only A's, either another A or the first B is possible according to the generative process.
- Loss: **Negative Log-Likelihood (NLL)** between the network's predicted log-probabilities and the target distribution.
- Models are saved every 10,000 epochs as `net/{lang}_{model}{hid}_{epoch//1000}.pth`.

**What we look for:**

After successful training, the hidden states for test sequences (n = 1, 2, 3, 4) are extracted and visualised. In a well-trained model:

- Points that share the same *n* (counter value) cluster together.
- Trajectories spiral or rotate around an origin in a way that systematically encodes the count.
- LSTM cell vectors often show a cleaner linear separation than the hidden (output) states because the cell state is unbounded while tanh squashes hidden states to `(-1, 1)`.

---

### 2. Binary Classification (MLP)

A small 2-D → hidden → 1-D MLP (`check.py`) is trained on a tabular dataset (`check.csv`) containing two continuous input features and a binary label. The training loop (`check_main.py`) runs until **100% training accuracy** is reached and then saves:

- `graph_hidden_{j}.jpg` — heat-map of each hidden unit's activation across the 2-D input space.
- `graph_output.jpg` — the network's output probability across the 2-D input space (decision boundary).

An alternative `set_weights()` method in `MLP` lets you manually specify weights for analysing specific Boolean functions (e.g., XOR), making this an excellent pedagogical tool for understanding how hidden units partition the feature space.

---

### 3. Image Classification (KMNIST / CNN)

Three progressively more powerful architectures (`kuzu.py`) are trained on the [KMNIST](https://github.com/rois-codh/kmnist) dataset (10 classes of Japanese hiragana characters, 28×28 grayscale images):

| Network | Architecture | Key Idea |
|---|---|---|
| `NetLin` | Flatten → Linear(784→10) → log_softmax | Linear baseline |
| `NetFull` | Flatten → Linear(784→150) → tanh → Linear(150→10) → log_softmax | Fully connected with non-linearity |
| `NetConv` | Conv(1→20, 3×3)+ReLU → MaxPool(4×4,s=2) → Conv(20→40, 3×3)+ReLU → MaxPool(4×4,s=2) → Linear(640→144)+ReLU → Linear(144→10) → log_softmax | 2-layer CNN |

Training uses SGD and `kuzu_main.py` reports:
- Per-epoch training loss
- Test accuracy
- A **full 10×10 confusion matrix** showing per-class predictions vs. ground truth

The dataset is automatically downloaded via `torchvision.datasets.KMNIST`.

---

## Model Architectures

### Simple Recurrent Network (SRN)

```
Input x_t (one-hot, D-dim)
      │
      ▼
  c_t = W·x_t + U·h_{t-1} + b_h      (pre-activation)
      │
   tanh(·)
      │
      ▼
   h_t   ──►  V·h_t + b_o  ──►  output y_t
      │
      └────────────────────────── recycled to next step
```

**Parameters:**
- `W` : input → hidden weight matrix `(D × H)`
- `U` : hidden → hidden (recurrent) weight matrix `(H × H)`
- `V` : hidden → output weight matrix `(H × D)`
- Learnable initial hidden state `H0` (passed through tanh)

### Long Short-Term Memory (LSTM)

```
Input x_t (one-hot, D-dim)
      │
      ▼
  gates = W·x_t + U·h_{t-1} + b        (shape: H×4)
      │
      ├─ i_t = σ(gates[:H])             input gate
      ├─ f_t = σ(gates[H:2H])           forget gate
      ├─ g_t = tanh(gates[2H:3H])       new values (cell update)
      └─ o_t = σ(gates[3H:])            output gate
      │
      ▼
  c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t   cell (context) state
      │
  h_t = o_t ⊙ tanh(c_t)                hidden state
      │
      ▼
  V·h_t + b_o  ──►  output y_t
```

**Key addition vs. SRN:** the cell state `c_t` is an additional information pathway that bypasses the squashing operations, allowing gradients and values to flow across long sequences. This project explicitly extracts `c_t` at every step (the "context sequence") for visualisation alongside `h_t`.

**Weight initialisation:** uniform distribution in `[-1/√H, +1/√H]` (Kaiming-style).

### Multi-Layer Perceptron (MLP)

```
Input (2-D features)
      │
  Linear(2 → H)
      │
  sigmoid  (or step)
      │
  Linear(H → 1)
      │
  sigmoid  (or step)
      │
  Binary output (0 / 1)
```

Default `H = 4`. The `set_weights()` method pre-loads hand-crafted weights for a specific Boolean function demonstration.

### Image Classification Networks

#### NetLin
```
28×28 image ─► Flatten(784) ─► Linear(784, 10) ─► LogSoftmax
```

#### NetFull
```
28×28 image ─► Flatten(784) ─► Linear(784, 150) ─► tanh ─► Linear(150, 10) ─► LogSoftmax
```

#### NetConv
```
28×28 × 1
  │
  Conv2d(1→20, kernel=3×3, stride=1)  ─► ReLU  ──►  26×26×20
  │
  MaxPool2d(kernel=4×4, stride=2)                 ──►  12×12×20
  │
  Conv2d(20→40, kernel=3×3, stride=1) ─► ReLU  ──►  10×10×40
  │
  MaxPool2d(kernel=4×4, stride=2)                 ──►  4×4×40
  │
  Flatten(640)
  │
  Linear(640, 144) ─► ReLU
  │
  Linear(144, 10)
  │
  LogSoftmax
```

---

## Datasets

| Dataset | Source | Task | Size | Format |
|---|---|---|---|---|
| `aⁿb²ⁿ` sequences | Generated by `anb2n.py` | Sequence prediction | Unlimited (on-the-fly) | One-hot tensors |
| `aⁿb²ⁿc³ⁿ` sequences | Generated by `anb2n.py` | Sequence prediction | Unlimited (on-the-fly) | One-hot tensors |
| `check.csv` | Local file (not tracked) | 2-D binary classification | Small tabular | CSV (2 features + label) |
| KMNIST | Auto-downloaded via torchvision | 10-class image classification | 60,000 train / 10,000 test | 28×28 grayscale PNG |

**Synthetic Language Format (anb2n):**

Each call to `get_sequence()` produces a batch of 5 concatenated language samples with random lengths `n ∈ [1, max_length]`. The return values are:

| Variable | Shape | Description |
|---|---|---|
| `input` | `(1, T-1, num_class)` | One-hot encoded input tokens |
| `seq` | `(T,)` | Integer token indices |
| `target` | `(1, T-1, num_class)` | Target probability distributions |
| `state` | list of T floats | Ground-truth counter state at each step |

---

## Dependencies & Installation

### Requirements

```
Python >= 3.8
torch >= 1.9
torchvision >= 0.10
numpy
pandas
matplotlib
scikit-learn
```

### Install

```bash
# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

> **No `requirements.txt` is currently committed.** You can generate one after installing:
> ```bash
> pip freeze > requirements.txt
> ```

---

## Usage

### Training Sequence Models (SRN / LSTM)

```bash
python seq_train.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--lang` | `anb2n` | Language: `anb2n` or `anb2nc3n` |
| `--length` | `4` | Maximum value of n (sequence complexity) |
| `--model` | `srn` | Architecture: `srn` or `lstm` |
| `--hid` | auto | Number of hidden units (default: 2 for `anb2n`/SRN, 3 for `anb2nc3n`/LSTM) |
| `--optim` | `sgd` | Optimiser: `sgd` or `adam` |
| `--lr` | `0.005` | Learning rate |
| `--mom` | `0` | SGD momentum |
| `--init` | `0.001` | Initial weight scale (SRN only) |
| `--epoch` | auto | Training epochs in thousands (default: 100k for `anb2n`, 200k for `anb2nc3n`/SRN) |
| `--out_path` | `net` | Directory to save `.pth` checkpoint files |

**Examples:**

```bash
# Train a 2-hidden-unit SRN on anb2n for 100,000 epochs
python seq_train.py --lang anb2n --model srn --hid 2

# Train a 3-hidden-unit LSTM on anb2nc3n with Adam optimiser
python seq_train.py --lang anb2nc3n --model lstm --hid 3 --optim adam --lr 0.001

# Train with increased sequence length and more epochs
python seq_train.py --lang anb2n --model lstm --hid 2 --length 6 --epoch 200
```

Every 1,000 epochs the script prints a training snapshot including:
- The generated symbol string (e.g., `AABBAAABBBBAABB`)
- Per-step hidden activations and context vectors (LSTM)
- Per-step predicted output probabilities
- Current epoch and mean squared error loss

Checkpoints are saved every 10,000 epochs.

---

### Visualising Sequence Model Trajectories

```bash
python seq_plot.py
```

This script loads a trained model from the `net/` directory (the default is `anb2n_srn2_100.pth`), runs it on a set of test sequences (n = 1…4), and saves scatter plots to JPEG files. With 3 or more hidden units, 3-D plots are also generated.

---

### Training the MLP Binary Classifier

```bash
python check_main.py
```

The script:
1. Loads `check.csv` (must be present in the working directory).
2. Trains until 100% accuracy or a maximum number of epochs.
3. Saves hidden-unit activation heat-maps as `graph_hidden_0.jpg`, `graph_hidden_1.jpg`, …
4. Saves the output decision boundary as `graph_output.jpg`.

---

### Training the KMNIST Image Classifier

```bash
python kuzu_main.py
```

KMNIST is automatically downloaded to `./data/`. The script trains for a fixed number of epochs, printing loss and accuracy after each epoch, then outputs the test confusion matrix.

To switch between network architectures, edit `kuzu_main.py` and change the instantiated class to `NetLin`, `NetFull`, or `NetConv`.

---

### Plotting Hidden-State Trajectories

Two standalone scripts visualise pre-computed (hard-coded) trajectory data for an LSTM trained on `aⁿb²ⁿc³ⁿ`:

```bash
# 3-D scatter plot of hidden states for n=1,2,3,4
python hidden_traj_all.py

# 3-D scatter plot of LSTM cell (context) states for n=1,2,3,4
python context_traj_all.py

# Multi-panel figure: 3-D + 3 × 2-D projections per sequence
python plot_traj.py
```

Each script produces and displays an interactive Matplotlib figure. Colour coding:

| Colour | Meaning |
|---|---|
| Blue | Token A |
| Orange | Token B |
| Green | Token C |
| Red line | Sequence n=1 |
| Yellow line | Sequence n=2 |
| Purple line | Sequence n=3 |
| Brown line | Sequence n=4 |

---

### Parsing Training Output

If you have captured the stdout of a training run to a text file, `text_to_dict.py` can convert it into a Python dictionary suitable for programmatic analysis or plotting:

```bash
python text_to_dict.py
```

Edit the `text` variable inside the script to paste in the captured output. The script uses regex to extract token labels, hidden-state vectors, and context vectors for each step and prints a structured dictionary.

---

## Key Results & Visualisations

### Hidden-State Spirals (SRN on `aⁿb²ⁿ`)

A well-trained 2-hidden-unit SRN encodes the counting variable `n` geometrically: the hidden state after processing `n` A characters lies at approximately the same angular position on a unit circle, scaled by `n`. The network has learned to represent integer counting with a 2-D spiral trajectory.

### LSTM Cell-State Linearity (`aⁿb²ⁿ`)

For the same task, an LSTM's cell state often shows a strikingly clean **linear** separation: after processing `n` A's the cell state lies approximately at position `n × v` along a fixed direction vector `v`. This is because the cell state is not squashed by tanh and can grow proportionally to `n`.

### Context vs. Hidden States (`aⁿb²ⁿc³ⁿ`)

With three character classes the 3-D visualisations reveal that:
- Hidden states cluster by character type (A / B / C regions are separated).
- Context vectors more clearly encode the *count* as their primary axis of variation.

### MLP Decision Boundary

The visualised hidden-unit activations show that each sigmoid hidden unit has learned a half-plane boundary (hyperplane in 2-D). The combination of these boundaries carves the input space into the correct regions for the target binary label.

### KMNIST Confusion Matrix

The convolutional network (`NetConv`) typically achieves > 90% test accuracy on KMNIST. The confusion matrix reveals which character classes are most often confused, guiding further model development.

---

## Technical Deep Dive

### Hidden State Dynamics

At each time step `t` the recurrent hidden state `h_t` is a point in an H-dimensional real-valued space. The **trajectory** `{h_0, h_1, …, h_T}` is the path this point traces as the network processes a sequence. For a network trained to accept `aⁿb²ⁿ`, the trajectory during the A-phase must encode `n` (the counter), and during the B-phase it must count down. This project makes those abstract dynamics concrete and visible.

### LSTM Context (Cell) Vectors

Unlike the hidden state `h_t` — which is squashed through `tanh(c_t) ⊙ o_t` — the cell state `c_t` is updated via a **linear recurrence**:

```
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
```

When the forget gate is close to 1 and the input gate close to 0 (during the B phase), `c_t ≈ c_{t-1}` and the cell acts as a **latch** holding the count. When the input gate is active (A phase) the cell accumulates. This makes the cell state a much more direct representation of the integer counter than the hidden state, and the visualisations in `context_traj_all.py` confirm this.

### Backpropagation Through Time (BPTT)

Training recurrent models uses **BPTT**: the network is unrolled across all time steps in the sequence and gradients are computed by standard reverse-mode automatic differentiation (PyTorch autograd). The loss at each step contributes to the weight updates, allowing the network to learn dependencies that span arbitrary distances in the sequence.

Weight decay (`weight_decay=0.0001`) is applied to prevent over-fitting and to encourage small-magnitude weights that generalise better across different values of `n`.

---

## File-by-File Reference

| File | Lines | Purpose |
|---|---|---|
| `seq_models.py` | 108 | `SRN_model` and `LSTM_model` PyTorch `nn.Module` subclasses |
| `seq_train.py` | 112 | CLI training loop for sequence models; saves `.pth` checkpoints |
| `seq_plot.py` | 112 | Loads checkpoint, extracts trajectories, saves scatter-plot JPEGs |
| `anb2n.py` | 81 | `lang_anb2n` class: generates `aⁿb²ⁿ` / `aⁿb²ⁿc³ⁿ` training batches |
| `check.py` | 43 | `MLP` class: 2-hidden-layer binary classifier with optional hand-set weights |
| `check_main.py` | 173 | Trains MLP, plots hidden-unit heat-maps and output decision boundary |
| `kuzu.py` | 93 | `NetLin`, `NetFull`, `NetConv`: three KMNIST classifiers |
| `kuzu_main.py` | 105 | Trains KMNIST classifier, prints confusion matrix |
| `plot_traj.py` | 148 | 4-panel trajectory figure (3-D + 3 × 2-D projections) |
| `hidden_traj_all.py` | 121 | 3-D trajectory plot for hidden states, all sequences |
| `context_traj_all.py` | 123 | 3-D trajectory plot for LSTM cell states, all sequences |
| `text_to_dict.py` | 102 | Regex parser: converts training log text into a Python dict |

---

## Configuration & Hyperparameters

### Default Hyperparameters

| Task | Model | Hidden Units | Epochs | Optimiser | LR |
|---|---|---|---|---|---|
| `anb2n` | SRN | 2 | 100k | SGD | 0.005 |
| `anb2n` | LSTM | 2 | 100k | SGD | 0.005 |
| `anb2nc3n` | SRN | 4 | 200k | SGD | 0.005 |
| `anb2nc3n` | LSTM | 3 | 100k | SGD | 0.005 |
| KMNIST | NetConv | — | 10 | SGD | 0.005 |
| Binary clf | MLP | 4 | Until 100% | SGD | 0.005 |

### Weight Initialisation

| Model | Method |
|---|---|
| SRN | `Normal(0, init)` where `init=0.001` by default |
| LSTM | `Uniform(-1/√H, +1/√H)` (Kaiming uniform) |
| MLP | PyTorch default (Kaiming uniform via `nn.Linear`) |
| CNN | PyTorch default (Kaiming uniform via `nn.Conv2d`) |

---

## Extending the Project

**Add a new formal language:**

1. Extend the `lang_anb2n` class in `anb2n.py` to generate the desired sequences.
2. Update `seq_train.py` to handle the new `--lang` option.
3. Specify the alphabet size via `num_class`.

**Try a GRU:**

Implement a `GRU_model` in `seq_models.py` following the same interface as `LSTM_model` (return `hidden_seq, output` — GRUs have no separate cell state). The training and plotting scripts will work with minimal modifications.

**Higher-dimensional visualisation:**

For models with more than 3 hidden units, use PCA or t-SNE (via `sklearn.decomposition.PCA` or `sklearn.manifold.TSNE`) to project trajectories to 3-D before plotting:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
reduced = pca.fit_transform(hidden_np.reshape(-1, H))
```

**Export trajectory data:**

Replace the hard-coded `DATA` dictionaries in `plot_traj.py`, `hidden_traj_all.py`, and `context_traj_all.py` with calls to `seq_plot.py`'s trajectory extraction logic so that plots always reflect the most recently trained model.

---

## Academic Context

This project was developed as **Assignment 1 for COMP9444 — Neural Networks and Deep Learning** at the **School of Computer Science and Engineering (CSE), University of New South Wales (UNSW)**. The assignment specification is included in `a1.pdf`.

COMP9444 covers:

- Feedforward and convolutional networks
- Recurrent networks and sequence modelling
- Representational learning and interpretability
- Formal language theory and its connections to RNN expressiveness

The project is a practical complement to theoretical results on the computational power of RNNs (Siegelmann & Sontag, 1995; Hochreiter & Schmidhuber, 1997).

> **Academic Integrity Notice:** This repository contains coursework material from COMP9444 at UNSW. If you are currently enrolled in this course, you must not copy or adapt any part of this code as your own submission. Refer to your institution's academic integrity policy before using this material.

---

*For questions about the code or experiments, please open a GitHub Issue.*
