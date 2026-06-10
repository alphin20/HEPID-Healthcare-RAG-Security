import json
import torch
import random
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------------
# Load dataset
# -------------------------------
with open("data/crafted_instruction_data_medquad.json") as f:
    data = json.load(f)

# -------------------------------
# Train-Test Split (80/20)
# -------------------------------
random.shuffle(data)

split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

print("Train size:", len(train_data))
print("Test size:", len(test_data))

# -------------------------------
# Dataset class
# -------------------------------
class DetectionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.samples = []

        for item in data:
            text = item["input"]
            label = item["label"]

            enc = tokenizer(
                text,
                truncation=True,
                max_length=512,
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

# -------------------------------
# Setup device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load tokenizer + model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(device)

# -------------------------------
# Create datasets
# -------------------------------
train_dataset = DetectionDataset(train_data, tokenizer)
test_dataset = DetectionDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -------------------------------
# Optimizer & Loss
# -------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(3):

    model.train()  # training mode
    total_loss = 0
    correct = 0
    total = 0

    print(f"\n🔵 Epoch {epoch}")

    for ids, mask, labels in tqdm(train_loader):

        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        outputs = model(ids, attention_mask=mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

    # -------------------------------
    # Evaluation (TEST)
    # -------------------------------
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for ids, mask, labels in test_loader:

            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

# -------------------------------
# Save model
# -------------------------------
os.makedirs("ckpt", exist_ok=True)

model.save_pretrained("ckpt")
tokenizer.save_pretrained("ckpt")

print("\n✅ Model saved to ./ckpt")