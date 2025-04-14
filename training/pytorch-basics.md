# Simple Torch

> `Data`, `Model`, `Loss & Optimizer`, `Training Loop`, and `Inference`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ===== 1. Data Preparation =====
# Create synthetic data (features and labels)
X = torch.randn(100, 3)  # 100 samples, 3 features
y = torch.randint(0, 2, (100,))  # Binary labels (0 or 1)

# Wrap in Dataset and DataLoader (for batching/shuffling)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ===== 2. Model Definition =====
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)  # Input: 3 features → Hidden: 64 units
        self.fc2 = nn.Linear(64, 1)  # Hidden → 1 output (binary classification)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for probability [0, 1]
        return x

model = SimpleNN()

# ===== 3. Training Setup =====
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# ===== 4. Training Loop =====
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Enable training mode (for dropout/batchnorm if used)
    total_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y.float())  # Squeeze to match shapes
        
        # Backward pass and optimize
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        total_loss += loss.item()
    
    # Print epoch stats
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ===== 5. Inference (Optional) =====
model.eval()  # Switch to evaluation mode
with torch.no_grad():
    test_input = torch.randn(1, 3)  # Single test sample
    prediction = model(test_input)
    print(f"Prediction probability: {prediction.item():.4f}")
```
