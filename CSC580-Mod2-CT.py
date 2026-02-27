# CSC580 Applying Machine Learning and Neural Networks
# Module 2 - Critical Thinking
# 
# Video Game Revenue Prediction
# Develop/train a neural network designed to predict future revenues from new video game sales.

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. Load Data
train_df = pd.read_csv(r'C:\Users\justi\Downloads\sales_data_training.csv')

# 2. Scale Data (9 inputs, 1 output)
X_train = train_df.drop('total_earnings', axis=1).values
y_train = train_df[['total_earnings']].values

X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = X_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

# Create DataLoader (shuffle=True)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define the Sequential Model
model = nn.Sequential(
    nn.Linear(9, 50),   # Input layer (9 inputs to 50 neurons)
    nn.ReLU(),
    nn.Linear(50, 20),  # Hidden layer (50 neurons to 20 neurons)
    nn.ReLU(),
    nn.Linear(20, 1)    # Output layer (20 neurons to 1 output)
)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 4. Training Loop (50 Epochs, Verbose=2 style)
print("Starting training...")
for epoch in range(1, 51):
    epoch_loss = 0.0
    model.train() # Set model to training mode
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Replicating Keras 'verbose=2'
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch}/50 - loss: {avg_loss:.4f}")

print("Training Complete.\n")

# 5. Evaluation with Test Dataset
test_df = pd.read_csv(r'C:\Users\justi\Downloads\sales_data_test.csv')

X_test_raw = test_df.drop('total_earnings', axis=1).values
y_test_raw = test_df[['total_earnings']].values

# Scale the test features and target using the TRAIN scalers
X_test_scaled = X_scaler.transform(X_test_raw)
y_test_scaled = y_scaler.transform(y_test_raw)

# Convert to PyTorch Tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

model.eval() # Sets the model to evaluation mode
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    
    # Calculate Mean Squared Error on test data
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"Test MSE Loss: {test_loss.item():.4f}")

    # 6. Convert back to Dollars
    final_predictions = y_scaler.inverse_transform(test_predictions.numpy())
    
    print("\n--- Sample Predictions (3 New Games) ---")
    for i in range(3):
        print(f"Actual: ${y_test_raw[i][0]:,.2f} | Predicted: ${final_predictions[i][0]:,.2f}")