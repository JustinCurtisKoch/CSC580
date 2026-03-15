# CSC580 Applying Machine Learning and Neural Networks
# Module 4 - Portfolio Milestone

# Predicting Fuel Efficiency
# Regression problem - predict the output of a continuous value (like a price or a probability).
# Classification problem - select a class from a list of classes (a picture contains an apple or an orange).
# Use the classic Auto MPG Dataset
# Builds a model to predict the fuel efficiency of late-1970s and early-1980s automobiles.
# The data model includes descriptions of many automobiles from that time period.
# The description for an automobile includes attributes such as cylinders, displacement, horsepower, and weight.
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

# --- 1. Data Prep (Same as before, but creating a Validation set) ---
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.dropna().copy()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Split: 80% Train, 20% Test
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Pop labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_dataset)
test_scaled = scaler.transform(test_dataset)

# Split Train into Train / Validation (80/20) for Early Stopping
X_train_full, X_val, y_train_full, y_val = train_test_split(
    train_scaled, train_labels.values, test_size=0.2, random_state=42)

# Convert to Tensors
X_train = torch.tensor(X_train_full, dtype=torch.float32)
y_train = torch.tensor(y_train_full, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(test_scaled, dtype=torch.float32)
y_test = torch.tensor(test_labels.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 2. Model Setup
class MPGRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MPGRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x)

model = MPGRegressor(X_train.shape[1])
criterion = nn.MSELoss()  
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# 3. Early Stopping Loop
epochs = 1000
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_weights = None

history = {'train_mae': [], 'val_mae': []}

print("Starting training with Early Stopping...")
for epoch in range(epochs):
    # Training step
    model.train()
    batch_mae = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_mae += torch.abs(outputs - y_batch).mean().item()
    
    train_mae = batch_mae / len(train_loader)
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, y_val).item()
        val_mae = torch.abs(val_preds - y_val).mean().item()
    
    history['train_mae'].append(train_mae)
    history['val_mae'].append(val_mae)
    
    # Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_weights = copy.deepcopy(model.state_dict()) # Save the best model
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered at Epoch {epoch+1}!")
        break

# Load best model weights back in
model.load_state_dict(best_model_weights)

# 4. Plot 1: Early Stopping History
print("\nClose the plot to continue...")
plt.figure(figsize=(8, 6))
plt.plot(history['train_mae'], label='Train MAE')
plt.plot(history['val_mae'], label='Validation MAE')
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.xlabel('Epoch')
plt.legend()
plt.title('Early Stopping Training History')
plt.show()

# 5. Print Test Set Eval
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_mae = torch.abs(test_predictions - y_test).mean().item()

print("\n" + "="*50)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(test_mae))
print("="*50 + "\n")

# 6. Plot 2: True vs Predictions Scatter
print("Close the plot to continue...")
test_preds_flat = test_predictions.numpy().flatten()
test_labels_flat = y_test.numpy().flatten()

plt.figure(figsize=(6, 6))
a = plt.axes(aspect='equal')
plt.scatter(test_labels_flat, test_preds_flat)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, color='red', linestyle='--')
plt.title('True vs Predicted MPG')
plt.show()

# 7. Plot 3: Error Distribution Histogram
print("Close the plot to finish...")
error = test_preds_flat - test_labels_flat
plt.figure(figsize=(8, 6))
plt.hist(error, bins=25, edgecolor='black')
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.title('Error Distribution')
plt.show()