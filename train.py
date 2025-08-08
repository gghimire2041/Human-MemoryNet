import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ToyConceptDataset
from model import MemoryNet

# Prepare dataset and loader
dataset = ToyConceptDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
model = MemoryNet(vocab_size=len(dataset), embedding_dim=8, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(2):
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "memorynet.pth")
print("Model saved as memorynet.pth")
