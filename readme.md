# Human MemoryNet: Neural Network Architecture Inspired by Human Memory 🧠
[Click here for a Quick Demo](https://gghimire2041.github.io/Human-MemoryNet/)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced neural network architecture that simulates human memory processes including encoding, storage, retrieval, and consolidation. This enhanced version is optimized for MacBook Pro 2018 with 16GB RAM and includes comprehensive visualizations and educational materials.

## 🎯 Overview

Human MemoryNet mimics the three fundamental stages of human memory:

1. **Encoding Stage**: Transform sensory input into neural representations
2. **Storage Stage**: Store memories in an external learnable memory matrix
3. **Retrieval Stage**: Use attention mechanisms to recall relevant memories

The architecture simulates both **episodic memory** (personal experiences) and **semantic memory** (factual knowledge), incorporating realistic human memory phenomena such as:

- 🔄 Memory consolidation over time
- 📉 Forgetting curves and memory decay
- 🎯 Selective attention and retrieval
- 💭 Context-dependent recall
- 🌟 Emotional influence on memory strength

## 🏗️ Architecture

### Core Components

```
Input → Embedding → Positional Encoding → Transformer Encoder
                                              ↓
External Memory ← Attention Mechanism ← Encoded Sequence
     ↓                    ↓
Memory Storage    →  Memory Retrieval  →  Context Integration
                                              ↓
                                        Output Layers
                                    (Classification, Prediction, Reconstruction)
```

### Key Features

- **External Memory Module**: Learnable memory matrix with 1000 slots (optimized for MacBook Pro)
- **Multi-Head Attention**: Selective memory retrieval mechanism
- **Temporal Context Processing**: Incorporates time-of-day, recency, and frequency information
- **Memory Consolidation**: Adaptive memory strengthening over time
- **Forgetting Mechanism**: Realistic memory decay simulation
- **Dual Memory Types**: Separate handling of episodic and semantic memories

## 📊 Model Specifications

| Component | Specification | Memory Usage |
|-----------|--------------|--------------|
| Vocabulary Size | 1,000 tokens | ~4 MB |
| Embedding Dimension | 128 | ~512 KB |
| Hidden Dimension | 256 | ~1 MB |
| Memory Slots | 1,000 | ~512 KB |
| Transformer Layers | 3 | ~2 MB |
| Total Parameters | ~380K | ~1.5 MB |
| **Peak Memory Usage** | **~100-200 MB** | **Fits comfortably in 16GB** |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gghimire2041/Human-MemoryNet.git
cd Human-MemoryNet

# Create virtual environment
python -m venv memory_env
source memory_env/bin/activate  # On Windows: memory_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from dataset import create_memory_dataloaders
from model import create_memory_model
from train import MemoryTrainer

# Create optimized dataloaders for MacBook Pro
train_loader, val_loader, test_loader = create_memory_dataloaders(
    train_size=3000,
    val_size=600,
    test_size=200,
    batch_size=16
)

# Create model optimized for MacBook Pro
model = create_memory_model()

# Train the model
config = {
    'epochs': 50,
    'batch_size': 16,
    'optimizer': {'type': 'adamw', 'lr': 1e-3}
}

trainer = MemoryTrainer(model, train_loader, val_loader, test_loader, config)
trainer.train(config['epochs'])
```

### Training

```bash
# Run full training pipeline
python train.py

# Monitor training with TensorBoard
tensorboard --logdir runs/
```

## 📁 Project Structure

```
Human-MemoryNet/
├── dataset.py              # Enhanced memory dataset with curated patterns
├── model.py                # Human MemoryNet architecture
├── train.py                # Training script with MacBook optimizations
├── requirements.txt        # Dependencies
├── README.md              # This file
├── visualization.html     # Interactive educational visualization
├── checkpoints/           # Saved model checkpoints
├── runs/                  # TensorBoard logs
└── examples/              # Usage examples and tutorials
    ├── basic_usage.py
    ├── memory_analysis.py
    └── visualization_demo.py
```

## 🎓 Educational Features

### Memory Types Simulated

#### 1. Episodic Memory
- **Personal experiences and events**
- Time-stamped sequences
- Context-dependent retrieval
- Emotional valence influence

Examples in dataset:
- Daily routines (morning, work, evening)
- Social events and interactions
- Learning experiences
- Travel memories

#### 2. Semantic Memory
- **Factual knowledge and concepts**
- Abstract relationships
- Category hierarchies
- Rule-based patterns

Examples in dataset:
- Mathematical concepts (sequences, operations)
- Language patterns (grammar, syntax)
- Scientific classifications
- General world knowledge

### Human Memory Phenomena Modeled

1. **Encoding Specificity**: Context-dependent memory formation
2. **Consolidation**: Memory strengthening over time
3. **Interference**: Competition between similar memories
4. **Forgetting Curve**: Exponential decay of unused memories
5. **Primacy/Recency Effects**: Enhanced recall for first/last items
6. **Emotional Enhancement**: Stronger encoding for emotional content

## 📈 Performance Metrics

The model tracks multiple metrics to evaluate human-like memory behavior:

- **Memory Type Accuracy**: Classification of episodic vs semantic memories
- **Memory Strength Prediction**: Estimation of
