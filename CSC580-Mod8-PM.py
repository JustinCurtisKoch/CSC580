# CSC580 Applying Machine Learning and Neural Networks
# Module 8 - Portfolio Milestone - Encoder-Decoder Model for Sequence-to-Sequence Prediction
# 
# Develop an encoder-decoder model for sequence-to-sequence prediction. 
# Encoder-decoder model is developed in PyTorch.
# Develop a sophisticated encoder-decoder RNN for a sequence-to-sequence prediction problem.

# There are three Python-based programming tasks for this Portfolio Project:
# 1. Encoder-Decoder Model in PyTorch
# 2. Scalable Sequence-to-Sequence Problem
# 3. Encoder-Decoder LSTM for Sequence Prediction.

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# Initialize hardware device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sequence(length, n_unique):
    """Generate a sequence of random integers within a specified range."""
    return [random.randint(1, n_unique - 1) for _ in range(length)]

def get_dataset(n_in, n_out, cardinality, n_samples):
    """Create a dataset of one-hot encoded tensors for training and evaluation."""
    X1, X2, Y = [], [], []
    for _ in range(n_samples):
        # Generate source and target sequences
        source = generate_sequence(n_in, cardinality)
        target = source[:n_out]
        target.reverse()
        target_in = [0] + target[:-1]
        
        # Convert to one-hot tensors
        src_tensor = torch.nn.functional.one_hot(torch.tensor(source), num_classes=cardinality).float()
        tar_in_tensor = torch.nn.functional.one_hot(torch.tensor(target_in), num_classes=cardinality).float()
        tar_tensor = torch.tensor(target) 
        
        X1.append(src_tensor)
        X2.append(tar_in_tensor)
        Y.append(tar_tensor)
        
    return torch.stack(X1).to(device), torch.stack(X2).to(device), torch.stack(Y).to(device)

def one_hot_decode(encoded_seq):
    """Transform one-hot encoded vectors back into integer representations."""
    return [torch.argmax(vector).item() for vector in encoded_seq]

class Encoder(nn.Module):
    """LSTM-based encoder module for sequence feature extraction."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        # Return final hidden and cell states
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    """LSTM-based decoder module for sequence generation."""
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        # Process input and produce prediction for the next time step
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

def predict_sequence(encoder, decoder, source_seq, n_steps, cardinality):
    """Generate a target sequence using recursive inference."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Encode source sequence
        hidden, cell = encoder(source_seq.unsqueeze(0))
        # Initialize start-of-sequence token
        target_seq = torch.zeros(1, 1, cardinality).to(device)
        output_seq = []
        
        for _ in range(n_steps):
            # Predict next element
            prediction, hidden, cell = decoder(target_seq, hidden, cell)
            predicted_idx = prediction.squeeze().argmax().item()
            output_seq.append(predicted_idx)
            
            # Prepare input for the next time step
            next_input = torch.zeros(1, 1, cardinality).to(device)
            next_input[0, 0, predicted_idx] = 1.0
            target_seq = next_input
            
    return output_seq

# Configuration parameters
n_features = 51
n_steps_in = 6
n_steps_out = 3
n_units = 128
epochs = 3000
batch_size = 32

# Model instantiation
encoder_model = Encoder(n_features, n_units).to(device)
decoder_model = Decoder(n_units, n_features).to(device)

# Optimization setup
optimizer = optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=0.005)
criterion = nn.CrossEntropyLoss()

# Training loop
losses = []
print("Initiating training sequence...")
encoder_model.train()
decoder_model.train()

for epoch in range(epochs):
    # Retrieve training batch
    X1_batch, X2_batch, y_batch = get_dataset(n_steps_in, n_steps_out, n_features, batch_size)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Forward pass
    hidden, cell = encoder_model(X1_batch)
    output, _, _ = decoder_model(X2_batch, hidden, cell)
    
    # Compute categorical loss
    loss = criterion(output.view(-1, n_features), y_batch.view(-1))
    losses.append(loss.item())
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Loss visualization
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True)
plt.legend()
plt.show()

# Final evaluation sequence
print("\nInitiating evaluation...")
total, correct = 100, 0
for _ in range(total):
    X1_test, _, y_test = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target_pred = predict_sequence(encoder_model, decoder_model, X1_test[0], n_steps_out, n_features)
    if target_pred == y_test[0].tolist():
        correct += 1

print(f'Final Accuracy: {float(correct)/float(total)*100.0:.2f}%')

# Sample output display
print("\nSample Results Visualization:")
for _ in range(10):
    X1_sample, _, y_sample = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    yhat_sample = predict_sequence(encoder_model, decoder_model, X1_sample[0], n_steps_out, n_features)
    print(f"X={one_hot_decode(X1_sample[0])} y={y_sample[0].tolist()}, yhat={yhat_sample}")