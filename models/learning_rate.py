import json
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# CONFIG
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS     = 3
MAX_LEN    = 512

# LOAD DATA
with open("data/crafted_instruction_data_medquad.json") as f:
    data = json.load(f)

random.shuffle(data)
data = data[:10000]

# 3-WAY SPLIT ← Fixed!
train_data = data[:8000]   # 80%
val_data   = data[8000:9000]  # 10%
test_data  = data[9000:]      # 10%

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# DATASET CLASS
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

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# LEARNING RATES
learning_rates = [1e-5, 2e-5, 5e-5]
results = []

for lr in learning_rates:
    print(f"\n{'='*30}\nTesting LR: {lr}\n{'='*30}")

    # Fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss()

    # Datasets
    train_loader = DataLoader(DetectionDataset(train_data, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(DetectionDataset(val_data,   tokenizer), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(DetectionDataset(test_data,  tokenizer), batch_size=BATCH_SIZE)

    # TRAINING + VALIDATION
    for epoch in range(EPOCHS):

        # Train
        model.train()
        total_train_loss = 0
        for ids, mask, labels in tqdm(train_loader):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            outputs = model(ids, attention_mask=mask)
            loss    = loss_fn(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                outputs = model(ids, attention_mask=mask)
                loss    = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # EVALUATION
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for ids, mask, labels in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            outputs   = model(ids, attention_mask=mask)
            preds     = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.numpy())

    results.append({
        "Learning Rate": lr,
        "Accuracy"     : accuracy_score(actuals, predictions),
        "Precision"    : precision_score(actuals, predictions),
        "Recall"       : recall_score(actuals, predictions),
        "F1 Score"     : f1_score(actuals, predictions)
    })

# RESULTS
df = pd.DataFrame(results)
print("\nFINAL LEARNING RATE COMPARISON")
print(df)
df.to_csv("learning_rate_comparison.csv", index=False)

# GRAPH
plt.figure(figsize=(8,5))
plt.plot(df["Learning Rate"].astype(str), df["Accuracy"], marker="o", label="Accuracy")
plt.plot(df["Learning Rate"].astype(str), df["F1 Score"], marker="s", label="F1 Score")
plt.xlabel("Learning Rate")
plt.ylabel("Score")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.savefig("learning_rate_comparison_graph.png")
plt.show()