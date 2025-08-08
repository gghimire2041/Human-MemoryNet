# MemoryNet Prototype

## Overview
This is a lightweight prototype of a neural network inspired by human memory organization:
- **Encoding Stage**: Input data is embedded into a vector space.
- **Storage Stage**: Vectors are stored in an external learnable memory matrix.
- **Retrieval Stage**: Attention mechanism retrieves relevant stored patterns.

The goal is to simulate aspects of *episodic* and *semantic* memory in a simplified setting.

---

## Files
- `dataset.py` – Toy dataset with simple concepts.
- `model.py` – MemoryNet model with an external memory module.
- `train.py` – Minimal training loop.
- `requirements.txt` – Dependencies.

---

## Running the Demo
```bash
pip install -r requirements.txt
python train.py
