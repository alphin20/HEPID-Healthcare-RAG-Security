import os
import json
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================
# DEVICE
# =========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# =========================================
# LOAD EXTERNAL DATASET
# =========================================
with open("new_external_dataset.json", "r") as f:
    data = json.load(f)

print("External test samples:", len(data))

texts = [x["input"] for x in data]
labels = [x["label"] for x in data]

# =========================================
# MODELS TO COMPARE
# =========================================
models = {
    "DistilBERT": "ckpt_distilbert",
    "BERT": "ckpt_bert",
    "RoBERTa": "ckpt_roberta"
}

results = []

# =========================================
# EVALUATION LOOP
# =========================================
for model_name, model_path in models.items():

    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    ).to(device)

    model.eval()

    predictions = []

    start_time = time.time()

    with torch.no_grad():

        for text in texts:

            encoding = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)

            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            pred = torch.argmax(
                outputs.logits,
                dim=1
            ).item()

            predictions.append(pred)

    end_time = time.time()

    inference_time = end_time - start_time

    # =====================================
    # METRICS
    # =====================================
    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions)

    recall = recall_score(labels, predictions)

    f1 = f1_score(labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1 Score: {f1:.4f}")

    print(f"Inference Time: {inference_time:.2f} sec")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Inference Time (s)": inference_time
    })

# =========================================
# SAVE RESULTS TABLE
# =========================================
df = pd.DataFrame(results)

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(df)

df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to model_comparison_results.csv")

# =========================================
# CREATE GRAPH
# =========================================
plt.figure(figsize=(10, 6))

plt.plot(
    df["Model"],
    df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    df["Model"],
    df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.xlabel("Models")

plt.ylabel("Score")

plt.title("Model Comparison")

plt.legend()

plt.grid(True)

plt.savefig(
    "model_comparison_graph.png"
)

print("Graph saved as model_comparison_graph.png")

plt.show()

print("\nComparison completed successfully.")