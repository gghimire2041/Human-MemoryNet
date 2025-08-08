
---

### **dataset.py**
```python
import torch
from torch.utils.data import Dataset

class ToyConceptDataset(Dataset):
    def __init__(self):
        # Toy data: simple "concept" IDs mapped to labels
        self.data = [
            ("apple", 0),
            ("banana", 0),
            ("dog", 1),
            ("cat", 1),
            ("car", 2),
            ("bus", 2)
        ]
        self.vocab = {word: i for i, (word, _) in enumerate(self.data)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word, label = self.data[idx]
        word_id = self.vocab[word]
        return torch.tensor(word_id), torch.tensor(label)
