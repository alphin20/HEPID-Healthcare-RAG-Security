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
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =========================================
# CONFIG
# =========================================
MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16

EPOCHS = 3
LR = 2e-5

MAX_LEN = 512

# =========================================
# LOAD DATASET
# =========================================
with open("data/crafted_instruction_data_medquad.json") as f:

    data = json.load(f)

random.shuffle(data)

# Use smaller subset for faster experiments
data = data[:10000]

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
# TOKENIZER
# =========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

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
# OPTIMIZERS TO TEST
# =========================================
optimizer_configs = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
    "SGD": torch.optim.SGD
}

results = []

# =========================================
# EXPERIMENT LOOP
# =========================================
for opt_name, opt_class in optimizer_configs.items():

    print(f"\n==============================")
    print(f"Testing Optimizer: {opt_name}")
    print(f"==============================")

    # Load fresh model every experiment
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)

    optimizer = opt_class(
        model.parameters(),
        lr=LR
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # =====================================
    # TRAINING
    # =====================================
    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for ids, mask, labels in tqdm(train_loader):

            ids = ids.to(device)

            mask = mask.to(device)

            labels = labels.to(device)

            outputs = model(
                ids,
                attention_mask=mask
            )

            loss = loss_fn(
                outputs.logits,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # =====================================
    # EVALUATION
    # =====================================
    model.eval()

    predictions = []

    actuals = []

    with torch.no_grad():

        for ids, mask, labels in test_loader:

            ids = ids.to(device)

            mask = mask.to(device)

            outputs = model(
                ids,
                attention_mask=mask
            )

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            actuals.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        actuals,
        predictions
    )

    print(f"{opt_name} Accuracy: {accuracy:.4f}")

    results.append({
        "Optimizer": opt_name,
        "Accuracy": accuracy
    })

# =========================================
# RESULTS TABLE
# =========================================
df = pd.DataFrame(results)

print("\n==============================")
print("FINAL OPTIMIZER COMPARISON")
print("==============================")

print(df)

# Save CSV
df.to_csv(
    "optimizer_comparison.csv",
    index=False
)

# =========================================
# GRAPH
# =========================================
plt.figure(figsize=(8,5))

plt.bar(
    df["Optimizer"],
    df["Accuracy"]
)

plt.xlabel("Optimizer")

plt.ylabel("Accuracy")

plt.title("Optimizer Comparison")

plt.grid(True)

plt.savefig(
    "optimizer_comparison_graph.png"
)

plt.show()

print("\nOptimizer experiment completed.")