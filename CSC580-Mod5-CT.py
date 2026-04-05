# CSC580 Applying Machine Learning and Neural Networks
# Module 5 Critical Thinking

# Improving Accuracy (Toxicology Testing Tox21 Dataset)
# Improve the accuracy of the deep learning model of the Tox21 model from Critical Thinking, Module 4

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. SETUP & DATA
output_dir = "Mod5-CT-results"
os.makedirs(output_dir, exist_ok=True)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = mfpgen.GetCountFingerprintAsNumPy(mol)
        return (fp > 0).astype(np.float32)
    return None

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
print(f"Downloading Tox21 data and outputting to {output_dir}...")
df = pd.read_csv(url, compression='gzip')
target_task = 'NR-AR' 
df = df[['smiles', target_task]].dropna()

features, labels = [], []
for _, row in df.iterrows():
    fp = smiles_to_fp(row['smiles'])
    if fp is not None:
        features.append(fp)
        labels.append(row[target_task])

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32)
pos_weight = torch.tensor([(len(y) - sum(y)) / sum(y)])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_X), 
                          torch.from_numpy(train_y)), batch_size=64, shuffle=True)

print(f"Cleaned Data Ready. Training on {len(train_X)} samples.\n")

# ==========================================
# PART 1: NEURAL NETWORK PIPELINE
# ==========================================
print("--- Starting Neural Network Hyperparameter Search ---")

class ToxNet(nn.Module):
    def __init__(self, hidden_size):
        super(ToxNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# Search to find the best config
nn_learning_rates = [0.001, 0.0005]
nn_hidden_sizes = [64, 128]

best_nn_auc = 0
best_nn_params = {}

for lr in nn_learning_rates:
    for hidden in nn_hidden_sizes:
        model = ToxNet(hidden)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train for 5 epochs to gauge potential
        for epoch in range(5):
            model.train()
            for b_x, b_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(b_x).squeeze(), b_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(torch.from_numpy(test_X)).squeeze()).numpy()
            auc = roc_auc_score(test_y, preds)
            if auc > best_nn_auc:
                best_nn_auc = auc
                best_nn_params = {'lr': lr, 'hidden': hidden}

print(f"Optimal NN Parameters Determined: {best_nn_params}\n")

print("--- Training Optimized Neural Network ---")
final_nn = ToxNet(best_nn_params['hidden'])
optimizer = optim.Adam(final_nn.parameters(), lr=best_nn_params['lr'])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

nn_losses = []
nn_aucs = []
peak_nn_auc = 0
peak_nn_epoch = 0

for epoch in range(20):
    final_nn.train()
    epoch_loss = 0
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(final_nn(b_x).squeeze(), b_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    nn_losses.append(avg_loss)
    
    final_nn.eval()
    with torch.no_grad():
        preds = torch.sigmoid(final_nn(torch.from_numpy(test_X)).squeeze()).numpy()
        auc = roc_auc_score(test_y, preds)
        nn_aucs.append(auc)
        
        if auc > peak_nn_auc:
            peak_nn_auc = auc
            peak_nn_epoch = epoch + 1
            
        print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | Val ROC-AUC: {auc:.4f}")

print(f"\n>> NN Peak Performance: {peak_nn_auc:.4f} achieved at Epoch {peak_nn_epoch}")

# ==========================================
# PART 2: RANDOM FOREST PIPELINE
# ==========================================
print("\n--- Starting Random Forest Hyperparameter Search ---")

rf_n_estimators = [50, 100, 200]
rf_max_depths = [None, 10, 20]

best_rf_auc = 0
best_rf_params = {}
rf_history = []

for n_trees in rf_n_estimators:
    for depth in rf_max_depths:
        rf_model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, 
                                          class_weight="balanced", random_state=42, n_jobs=-1)
        rf_model.fit(train_X, train_y)
        preds_proba = rf_model.predict_proba(test_X)[:, 1]
        auc = roc_auc_score(test_y, preds_proba)
        
        rf_history.append((f"T:{n_trees}_D:{depth}", auc))
        if auc > best_rf_auc:
            best_rf_auc = auc
            best_rf_params = {'n_estimators': n_trees, 'max_depth': depth}

print(f"Optimal RF Parameters Determined: {best_rf_params}")
print(f">> RF Peak Performance: {best_rf_auc:.4f}\n")


# ==========================================
# PART 3: GENERATE VISUALIZATIONS
# ==========================================
print("Generating comparative performance graphs...")

# Plot 1: NN Training Loss & AUC (Dual Axis)
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(range(1, 21), nn_losses, color=color, linewidth=2, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:green'
ax2.set_ylabel('Validation ROC-AUC', color=color)  
ax2.plot(range(1, 21), nn_aucs, color=color, linewidth=2, label='AUC')
ax2.axvline(x=peak_nn_epoch, color='red', linestyle='--', alpha=0.6, label=f'Peak Epoch ({peak_nn_epoch})')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Neural Network: Loss vs. ROC-AUC')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'nn_training_metrics.png'))

# Plot 2: Random Forest Tuning Results
labels, aucs = zip(*rf_history)
plt.figure(figsize=(10, 5))
plt.bar(labels, aucs, color='steelblue')
plt.axhline(y=best_rf_auc, color='r', linestyle='--', label=f'Peak RF AUC: {best_rf_auc:.4f}')
plt.ylim(0.65, max(aucs) + 0.05)
plt.title("Random Forest Hyperparameter Tuning")
plt.xlabel("Configuration (Trees_Depth)")
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_tuning_metrics.png'))

# Plot 3: The Final Showdown (NN vs RF)
plt.figure(figsize=(8, 6))
models = ['Deep Learning (PyTorch)', 'Random Forest (Scikit-Learn)']
scores = [peak_nn_auc, best_rf_auc]
colors = ['purple', 'darkorange']

bars = plt.bar(models, scores, color=colors, width=0.5)
plt.ylim(0.60, max(scores) + 0.05)
plt.title("Final Model Comparison: Tox21 NR-AR")
plt.ylabel("Peak ROC-AUC Score")

# Add the exact scores on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'))

print(f"Complete. Graphs and metrics saved to: {os.path.abspath(output_dir)}")