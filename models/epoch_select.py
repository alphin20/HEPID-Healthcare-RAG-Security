import json
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tqdm import tqdm

# =========================================
# CONFIG — Best params from previous experiments
# =========================================
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
MAX_EPOCHS = 10        # Test 1 to 10 epochs
LR         = 2e-5      # Best LR from previous experiment
MAX_LEN    = 512

# =========================================
# LOAD DATASET
# =========================================
with open("data/crafted_instruction_data_medquad.json") as f:
    data = json.load(f)

random.shuffle(data)
data = data[:10000]

# 3-WAY SPLIT
train_data = data[:8000]      # 80%
val_data   = data[8000:9000]  # 10%
test_data  = data[9000:]      # 10%

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# =========================================
# DATASET CLASS
# =========================================
class DetectionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.samples = []
        for item in data:
            enc = tokenizer(
                item["input"],
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt"
            )
            self.samples.append((
                enc["input_ids"].squeeze(),
                enc["attention_mask"].squeeze(),
                torch.tensor(item["label"], dtype=torch.long)
            ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# =========================================
# DEVICE & TOKENIZER
# =========================================
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Using device:", device)

# =========================================
# DATA LOADERS
# =========================================
train_loader = DataLoader(
    DetectionDataset(train_data, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    DetectionDataset(val_data, tokenizer),
    batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    DetectionDataset(test_data, tokenizer),
    batch_size=BATCH_SIZE
)

# =========================================
# FRESH MODEL
# =========================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn   = torch.nn.CrossEntropyLoss()

# =========================================
# TRACKING
# =========================================
train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []
results      = []

# =========================================
# TRAINING LOOP — 10 EPOCHS
# =========================================
for epoch in range(MAX_EPOCHS):
    print(f"\n{'='*30}")
    print(f"EPOCH {epoch+1}/{MAX_EPOCHS}")
    print(f"{'='*30}")

    # ── TRAIN ──
    model.train()
    total_train_loss = 0
    train_preds, train_actuals = [], []

    for ids, mask, labels in tqdm(train_loader):
        ids, mask, labels = (
            ids.to(device),
            mask.to(device),
            labels.to(device)
        )
        outputs = model(ids, attention_mask=mask)
        loss    = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_actuals.extend(labels.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc  = accuracy_score(train_actuals, train_preds)

    # ── VALIDATE ──
    model.eval()
    total_val_loss = 0
    val_preds, val_actuals = [], []

    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask, labels = (
                ids.to(device),
                mask.to(device),
                labels.to(device)
            )
            outputs = model(ids, attention_mask=mask)
            loss    = loss_fn(outputs.logits, labels)

            total_val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_actuals.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc  = accuracy_score(val_actuals, val_preds)

    # ── PRINT ──
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Train Acc:  {avg_train_acc:.4f}  | Val Acc:  {avg_val_acc:.4f}")

    # ── STORE ──
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)

    results.append({
        "Epoch"      : epoch + 1,
        "Train Loss" : avg_train_loss,
        "Val Loss"   : avg_val_loss,
        "Train Acc"  : avg_train_acc,
        "Val Acc"    : avg_val_acc
    })

# =========================================
# RESULTS TABLE
# =========================================
df = pd.DataFrame(results)
print("\nEPOCH EXPERIMENT RESULTS")
print(df.to_string(index=False))
df.to_csv("epoch_comparison.csv", index=False)

# =========================================
# OVERFITTING PROOF GRAPHS
# =========================================
epochs = range(1, MAX_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Plot
ax1.plot(epochs, train_losses, marker='o', label='Train Loss',     color='blue')
ax1.plot(epochs, val_losses,   marker='s', label='Val Loss',       color='orange')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Train vs Val Loss")
ax1.legend()
ax1.grid(True)

# Accuracy Plot
ax2.plot(epochs, train_accs, marker='o', label='Train Accuracy',   color='blue')
ax2.plot(epochs, val_accs,   marker='s', label='Val Accuracy',     color='orange')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.set_title("Train vs Val Accuracy")
ax2.legend()
ax2.grid(True)

plt.suptitle("Epoch Selection — Overfitting Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("epoch_comparison_graph.png")
plt.show()

# =========================================
# BEST EPOCH
# =========================================
best_epoch = df.loc[df['Val Acc'].idxmax(), 'Epoch']
print(f"\nEpoch experiment completed.")
print(f"Best Epoch → {best_epoch}")
