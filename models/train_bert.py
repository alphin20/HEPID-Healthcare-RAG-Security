import json
import torch
import random
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================================
# CONFIG
# =========================================
MODEL_NAME = "distilbert-base-uncased"

SAVE_DIR = "ckpt_distilbert"

BATCH_SIZE = 16

EPOCHS = 3

LR = 2e-5

MAX_LEN = 512

# =========================================
# LOAD DATASET
# =========================================
with open("data/crafted_instruction_data_medquad.json") as f:

    data = json.load(f)

# Shuffle dataset
random.shuffle(data)

# Reduce dataset size for faster training
data = data[:2000]

print("Total samples used:", len(data))

# =========================================
# TRAIN TEST SPLIT
# =========================================
split = int(0.8 * len(data))

train_data = data[:split]

test_data = data[split:]

print("Train size:", len(train_data))

print("Test size:", len(test_data))

# =========================================
# DATASET CLASS
# =========================================
class DetectionDataset(Dataset):

    def __init__(self, data, tokenizer):

        self.samples = []

        for item in data:

            text = item["input"]

            label = item["label"]

            enc = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt"
            )

            self.samples.append((
                enc["input_ids"].squeeze(),
                enc["attention_mask"].squeeze(),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]

# =========================================
# DEVICE
# =========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# =========================================
# TOKENIZER + MODEL
# =========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

# =========================================
# DATASETS
# =========================================
train_dataset = DetectionDataset(
    train_data,
    tokenizer
)

test_dataset = DetectionDataset(
    test_data,
    tokenizer
)

# =========================================
# DATALOADERS
# =========================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

# =========================================
# OPTIMIZER
# =========================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

# =========================================
# LOSS FUNCTION
# =========================================
loss_fn = torch.nn.CrossEntropyLoss()

# =========================================
# TRAINING LOOP
# =========================================
train_losses = []

train_accuracies = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for ids, mask, labels in tqdm(train_loader):

        ids = ids.to(device)

        mask = mask.to(device)

        labels = labels.to(device)

        # Forward pass
        outputs = model(
            ids,
            attention_mask=mask
        )

        loss = loss_fn(
            outputs.logits,
            labels
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(
            outputs.logits,
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)

    accuracy = correct / total

    train_losses.append(avg_loss)

    train_accuracies.append(accuracy)

    print(f"Loss: {avg_loss:.4f}")

    print(f"Accuracy: {accuracy:.4f}")

# =========================================
# SAVE MODEL
# =========================================
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)

tokenizer.save_pretrained(SAVE_DIR)

print("\nModel saved to:", SAVE_DIR)

# =========================================
# SAVE TRAINING LOGS
# =========================================
os.makedirs("logs", exist_ok=True)

with open("logs/distilbert_training_logs.json", "w") as f:

    json.dump({
        "loss": train_losses,
        "accuracy": train_accuracies
    }, f)

print("Training logs saved.")

print("\nTraining completed successfully.")