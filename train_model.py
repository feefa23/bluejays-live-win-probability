import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading dataset via Pandas...")
df = pd.read_csv("blue_jays_training_data.csv")

feature_columns = [
    'inning', 'is_blue_jays_batting', 'outs', 'balls', 'strikes',
    'runner_on_1st', 'runner_on_2nd', 'runner_on_3rd', 'score_differential',
    'batter_avg', 'batter_ops', 'pitcher_era', 'pitcher_so_rate'
]

# Hardcoded Min-Max bounds to equalize neural network feature scale weight values
feature_mins = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
feature_maxs = np.array([12.0, 1.0, 3.0, 4.0, 3.0, 1.0, 1.0, 1.0, 15.0, 0.400, 1.200, 15.0, 1.0], dtype=np.float32)

X_raw = df[feature_columns].values.astype(np.float32)
y_numpy = df['blue_jays_won'].values.astype(np.float32)

# Apply explicit Min-Max normalization formula scaling
X_scaled = (X_raw - feature_mins) / (feature_maxs - feature_mins)

X = torch.tensor(X_scaled)
y = torch.tensor(y_numpy).unsqueeze(1)

class WinProbabilityModel(nn.Module):
    def __init__(self):
        super(WinProbabilityModel, self).__init__()
        self.hidden = nn.Linear(13, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = WinProbabilityModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("Starting Balanced Feature Model Training Loop...")
for epoch in range(150):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 15 == 0:
        print(f"Epoch [{epoch + 1}/150] - Scaled Loss Rate: {loss.item():.4f}")

torch.save(model.state_dict(), "blue_jays_model.pth")
print("\nModel metrics training saved successfully to blue_jays_model.pth!")