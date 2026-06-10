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
EPOCHS     = 3       # Best epoch from previous experiment
LR         = 2e-5    # Best LR from previous experiment
MAX_LEN    = 512

# =========================================
# BATCH SIZES TO TEST
# =========================================
batch_sizes = [8, 16, 32, 64]

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
# RESULTS STORAGE
# =========================================
results        = []
all_train_loss = {}
all_val_loss   = {}
all_train_acc  = {}
all_val_acc    = {}

# =========================================
# EXPERIMENT LOOP
# =========================================
for bs in batch_sizes:
    print(f"\n{'='*30}")
    print(f"Testing Batch Size: {bs}")
    print(f"{'='*30}")

    # Data Loaders
    train_loader = DataLoader(
        DetectionDataset(train_data, tokenizer),
        batch_size=bs,
        shuffle=True
    )
    val_loader = DataLoader(
        DetectionDataset(val_data, tokenizer),
        batch_size=bs
    )
    test_loader = DataLoader(
        DetectionDataset(test_data, tokenizer),
        batch_size=bs
    )

    # Fresh Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = torch.nn.CrossEntropyLoss()

    # Tracking per batch size
    epoch_train_losses = []
    epoch_val_losses   = []
    epoch_train_accs   = []
    epoch_val_accs     = []

    # ── TRAINING LOOP ──
    for epoch in range(EPOCHS):

        # Train
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

        # Validate
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

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")

        epoch_train_losses.append(avg_train_loss)
        epoch_val_losses.append(avg_val_loss)
        epoch_train_accs.append(avg_train_acc)
        epoch_val_accs.append(avg_val_acc)

    # Store per batch tracking
    all_train_loss[bs] = epoch_train_losses
    all_val_loss[bs]   = epoch_val_losses
    all_train_acc[bs]  = epoch_train_accs
    all_val_acc[bs]    = epoch_val_accs

    # Final Evaluation on Test Set
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for ids, mask, labels in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            outputs   = model(ids, attention_mask=mask)
            preds     = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    results.append({
        "Batch Size" : bs,
        "Accuracy"   : accuracy_score(actuals, predictions),
        "Precision"  : precision_score(actuals, predictions),
        "Recall"     : recall_score(actuals, predictions),
        "F1 Score"   : f1_score(actuals, predictions)
    })

# =========================================
# RESULTS TABLE
# =========================================
df = pd.DataFrame(results)
print("\nFINAL BATCH SIZE COMPARISON")
print(df.to_string(index=False))
df.to_csv("batch_size_comparison.csv", index=False)

# =========================================
# GRAPHS
# =========================================
epochs = range(1, EPOCHS + 1)
colors = ['blue', 'orange', 'green', 'red']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graph 1 — Train Loss
for i, bs in enumerate(batch_sizes):
    axes[0,0].plot(epochs, all_train_loss[bs], marker='o', label=f'BS={bs}', color=colors[i])
axes[0,0].set_title("Train Loss per Batch Size")
axes[0,0].set_xlabel("Epochs")
axes[0,0].set_ylabel("Loss")
axes[0,0].legend()
axes[0,0].grid(True)

# Graph 2 — Val Loss
for i, bs in enumerate(batch_sizes):
    axes[0,1].plot(epochs, all_val_loss[bs], marker='s', label=f'BS={bs}', color=colors[i])
axes[0,1].set_title("Val Loss per Batch Size")
axes[0,1].set_xlabel("Epochs")
axes[0,1].set_ylabel("Loss")
axes[0,1].legend()
axes[0,1].grid(True)

# Graph 3 — Train Accuracy
for i, bs in enumerate(batch_sizes):
    axes[1,0].plot(epochs, all_train_acc[bs], marker='o', label=f'BS={bs}', color=colors[i])
axes[1,0].set_title("Train Accuracy per Batch Size")
axes[1,0].set_xlabel("Epochs")
axes[1,0].set_ylabel("Accuracy")
axes[1,0].legend()
axes[1,0].grid(True)

# Graph 4 — Final Accuracy Bar Chart
axes[1,1].bar(
    df["Batch Size"].astype(str),
    df["Accuracy"],
    color=colors
)
axes[1,1].set_title("Final Test Accuracy per Batch Size")
axes[1,1].set_xlabel("Batch Size")
axes[1,1].set_ylabel("Accuracy")
axes[1,1].grid(True)

plt.suptitle("Batch Size Selection Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("batch_size_comparison_graph.png")
plt.show()

# =========================================
# BEST BATCH SIZE
# =========================================
best_bs = df.loc[df['F1 Score'].idxmax(), 'Batch Size']
print(f"\nBatch Size experiment completed.")
print(f"Best Batch Size → {best_bs}")