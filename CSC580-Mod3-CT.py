# CSC580 Applying Machine Learning and Neural Networks
# Module 2 - Critical Thinking

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
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1 & 2: Download and Import Database
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values="?", 
                          comment='\t', sep=" ", skipinitialspace=True)

# Clean missing values
dataset = raw_dataset.dropna().copy()

# One-Hot Encode 'Origin'
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Step 3: tail of the dataset
print("\n" + "="*40)
print("--- Step 3: Tail of the Dataset ---")
print(dataset.tail())
print("="*40 + "\n")

# Step 4: Split the data into train and test
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Step 5 & 6: Inspect data
print("Generating Pairplot for Step 6. (Close the window to continue script...)")
sns.set_theme(style="whitegrid")
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# Step 7 & 8: Review statistics
train_stats = train_dataset.describe().transpose()
print("\n" + "="*40)
print("--- Step 8: Tail of the Statistics ---")
print(train_stats.tail())
print("="*40 + "\n")

# Step 9 & 10: Separate labels from features
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Step 11: Normalize the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_dataset)
test_scaled = scaler.transform(test_dataset)

# Convert arrays to PyTorch Tensors
X_train = torch.tensor(train_scaled, dtype=torch.float32)
y_train = torch.tensor(train_labels.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(test_scaled, dtype=torch.float32)
y_test = torch.tensor(test_labels.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Step 12: Build the model
class MPGRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MPGRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

model = MPGRegressor(X_train.shape[1])

# Step 13, 14 & 16: Inspect model and print summary
print("\n" + "="*40)
print("--- Step 14 & 16: Model Summary ---")
print(model)
print("="*40 + "\n")

# Step 15: Try model on a batch of 10
print("\n" + "="*40)
print("--- Step 15: Predictions on a batch of 10 examples ---")
sample_batch = X_train[:10]
sample_preds = model(sample_batch)
print(sample_preds.detach().numpy())
print("="*40 + "\n")

# Step 17 & 18: Train the model
criterion = nn.MSELoss()  
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

epochs = 1000
history = {'loss': [], 'mae': []}

print("Training the model for 1000 epochs...")
for epoch in range(epochs):
    model.train()
    batch_loss = 0
    batch_mae = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        batch_loss += loss.item()
        batch_mae += torch.abs(outputs - y_batch).mean().item()
    
    history['loss'].append(batch_loss / len(train_loader))
    history['mae'].append(batch_mae / len(train_loader))
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss(MSE): {history['loss'][-1]:.4f}, MAE: {history['mae'][-1]:.4f}")

# Step 19, 20 & 21: Visualize training progress
print("\nGenerating Training History Plot for Steps 20 & 21. (Close the window to continue script...)")
plt.figure(figsize=(8, 6))
plt.plot(history['mae'], label='MAE [MPG]')
plt.plot(history['loss'], label='MSE [MPG^2]')
plt.ylim([0, 20])
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training Progress')
plt.legend()
plt.show()

# Step 22: Compare models & Final Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_mae = torch.abs(predictions - y_test).mean()
    test_mse = criterion(predictions, y_test)

print("\n" + "="*40)
print("--- Step 22: Final Evaluation ---")
print(f"Test MAE:  {test_mae:.2f} MPG")
print(f"Test MSE:  {test_mse:.2f} MPG^2")
print("="*40 + "\n")

# Residual Analysis
print("Generating final Residual Analysis plot...")
residuals = y_test.numpy() - predictions.numpy()
plt.figure(figsize=(8, 6))
plt.scatter(predictions.numpy(), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()