import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Use PANDAS to load our spreadsheet
print("Loading training dataset via Pandas...")
df = pd.read_csv("blue_jays_training_data.csv")

# 2. Select our "Features" (the inputs our model looks at to make a guess)
feature_columns = ['inning', 'outs', 'balls', 'strikes', 'score_differential']

# Convert the Pandas columns into a raw NUMPY grid of numbers
X_numpy = df[feature_columns].values.astype(np.float32)

# Select our "Target" (what we want the model to predict: 1 for win, 0 for loss)
y_numpy = df['blue_jays_won'].values.astype(np.float32)

# 3. Convert NumPy grids into PYTORCH Tensors so the neural network can read them
X = torch.tensor(X_numpy)
y = torch.tensor(y_numpy).unsqueeze(1) # Reshapes the grid to match the model layout

print(f"Dataset successfully loaded. Matrix shape: {X.shape}")

# 4. Define the Brain (The Neural Network architecture)
class WinProbabilityModel(nn.Module):
    def __init__(self):
        super(WinProbabilityModel, self).__init__()
        # Input layer expects 5 numbers (inning, outs, balls, strikes, diff)
        # Hidden layer has 16 digital neurons
        self.hidden = nn.Linear(5, 16)
        # Output layer compresses everything down to 1 number (probability between 0 and 1)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.hidden(x)) # Activation function to find complex patterns
        x = self.sigmoid(self.output(x))
        return x

# Instantiate our live model
model = WinProbabilityModel()

# Define how the model calculates mistakes (Loss) and how it learns (Optimizer)
criterion = nn.BCELoss() # Binary Cross Entropy (standard for 0 or 1 guessing)
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimizer with a learning rate of 1%

print("\nStarting Model Training...")

# 5. The Training Loop (Let the model look at the 12,457 rows 100 times over to learn)
for epoch in range(100):
    # Reset tracking gradients
    optimizer.zero_grad()
    
    # Pass all data through the model to get its current guesses
    predictions = model(X)
    
    # Calculate how wrong the model's guesses were compared to actual game outcomes
    loss = criterion(predictions, y)
    
    # Backpropagation: Figure out how to tweak the digital neurons to make fewer mistakes
    loss.backward()
    optimizer.step()
    
    # Print progress every 10 loops
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100] - Loss (Mistake Rate): {loss.item():.4f}")

print("\nModel training complete!")

# 6. Save the trained brain to a file so we can use it live later!
torch.save(model.state_dict(), "blue_jays_model.pth")
print("Saved trained model parameters to blue_jays_model.pth")