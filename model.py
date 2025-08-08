import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):
    def __init__(self, memory_size, embedding_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
    
    def forward(self, query):
        attn_weights = F.softmax(torch.matmul(query, self.memory.T), dim=-1)
        retrieved = torch.matmul(attn_weights, self.memory)
        return retrieved

class MemoryNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, memory_size=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.memory_module = MemoryModule(memory_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        emb = self.embedding(x)
        retrieved = self.memory_module(emb)
        out = self.fc(retrieved)
        return out
