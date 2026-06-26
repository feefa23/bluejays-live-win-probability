import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Load our rich dataset via Pandas
print("Loading advanced situational dataset...")
df = pd.read_csv("blue_jays_training_data.csv")

# 2. Select our expanded 14 features
feature_columns = [
    'inning', 'is_blue_jays_batting', 'outs', 'balls', 'strikes',
    'runner_on_1st', 'runner_on_2nd', 'runner_on_3rd', 'score_differential',
    'batter_avg', 'batter_ops', 'pitcher_era', 'pitcher_so_rate'
]

# Convert the spreadsheet columns into a raw NumPy grid
X_numpy = df[feature_columns].values.astype(np.float32)
y_numpy = df['blue_jays_won'].values.astype(np.float32)

# Convert NumPy grids into PyTorch Tensors
X = torch.tensor(X_numpy)
y = torch.tensor(y_numpy).unsqueeze(1)

print(f"Dataset successfully loaded. Matrix shape: {X.shape} (14 features per play)")

# 3. Define the Expanded Brain
class WinProbabilityModel(nn.Module):
    def __init__(self):
        super(WinProbabilityModel, self).__init__()
        # Input layer now expects 13 numbers instead of 5
        self.hidden = nn.Linear(13, 32) # Bumped up to 32 digital neurons to handle complex player combinations
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Instantiate our model and optimization tools
model = WinProbabilityModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) # Adjusted learning rate for deeper pattern matching

print("\nStarting Advanced Model Training...")

# The Training Loop
for epoch in range(120): # Giving it 120 cycles to process the bigger matrix
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/120] - Loss (Mistake Rate): {loss.item():.4f}")

print("\nModel training complete!")

# Save the new brain weights
torch.save(model.state_dict(), "blue_jays_model.pth")
print("Saved advanced trained model parameters to blue_jays_model.pth")