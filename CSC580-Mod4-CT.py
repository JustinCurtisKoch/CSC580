import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. SETUP OUTPUT DIRECTORY
output_dir = "Mod4-CT-results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Silence RDKit noise
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Initialize Generator
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = mfpgen.GetCountFingerprintAsNumPy(mol)
        return (fp > 0).astype(np.float32)
    return None

# 2. Data Acquisition
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
print(f"Downloading Tox21 data and outputting to {output_dir}...")
df = pd.read_csv(url, compression='gzip')
target_task = 'NR-AR' 
df = df[['smiles', target_task]].dropna()

features, labels = [], []
for index, row in df.iterrows():
    fp = smiles_to_fp(row['smiles'])
    if fp is not None:
        features.append(fp)
        labels.append(row[target_task])

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Cleaned Data Ready! Training on {len(train_X)} samples.\n")

# 3. Model Architecture
class ToxModel(nn.Module):
    def __init__(self):
        super(ToxModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.net(x)

model = ToxModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y)), 
                          batch_size=64, shuffle=True)

losses = []
auc_scores = []

print("Starting Training...")
for epoch in range(15):
    model.train()
    epoch_loss = 0
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        output = model(b_x).squeeze()
        loss = criterion(output, b_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.from_numpy(test_X)).squeeze()).numpy()
        auc = roc_auc_score(test_y, preds)
        auc_scores.append(auc)
        print(f"Epoch {epoch+1:02d}/15 | Avg Loss: {avg_loss:.4f} | Val ROC-AUC: {auc:.4f}")

# 5. GENERATE AND SAVE PLOTS TO SUB-FOLDER
print("\nGenerating performance graphs...")

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.title('Model Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))

# Plot AUC
plt.figure(figsize=(10, 5))
plt.plot(auc_scores, label='Validation ROC-AUC', color='green', linewidth=2)
plt.title('Model ROC-AUC Performance')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(output_dir, 'auc_curve.png'))

print(f"Success! Graphs saved in: {os.path.abspath(output_dir)}")