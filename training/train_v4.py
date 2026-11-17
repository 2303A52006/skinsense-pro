"""Training and evaluation script for SkinSense Pro (final model, v4).

Run in Google Colab with a GPU after downloading HAM10000 into /content/data
(HAM10000_metadata.csv and the images). Saves /content/models/skinsense_model_v4.pth
and /content/models/eval_results_v4.json. Run it with:  python train_v4.py
"""


import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------- Settings you can change ----------
EPOCHS = 12
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
EXTRA_WEIGHT = {"mel": 1.5, "bcc": 1.5}  # extra attention on the serious cancers

DATA_DIR = "/content/data"
MODEL_OUT = "/content/models/skinsense_model_v4.pth"
EVAL_OUT = "/content/models/eval_results_v4.json"

# Numbers from the first model, printed at the end for comparison
BEFORE = {"balanced": 72.52, "mel": 64.0, "bcc": 54.4}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)
np.random.seed(42)

# 1. Data (same split as before, so the comparison is fair)
disease_names = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}
class_names = list(disease_names.keys())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))
df["label"] = df["dx"].map(class_to_idx)

image_files = glob.glob(DATA_DIR + "/**/*.jpg", recursive=True)
image_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
df["image_path"] = df["image_id"].map(image_map)
df = df.dropna(subset=["image_path"]).copy()

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(df, groups=df["lesion_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

overlap = len(set(train_df["lesion_id"]) & set(val_df["lesion_id"]))
print("Training images:", len(train_df), "| Validation images:", len(val_df))
print("Lesions in BOTH sets (should be 0):", overlap)

# 2. Image transforms (stronger augmentation for training only)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    normalize,
])

# Validation transform is unchanged, because the app uses the same one
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


class SkinDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["label"])


train_loader = DataLoader(
    SkinDataset(train_df, train_transform), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=True,
)
val_loader = DataLoader(
    SkinDataset(val_df, val_transform), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=True,
)

# 3. Model, loss and optimiser
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

class_counts = train_df["label"].value_counts().sort_index()
weights = (len(train_df) / (len(class_counts) * class_counts)).values.astype(np.float32)
for code, factor in EXTRA_WEIGHT.items():
    weights[class_to_idx[code]] *= factor
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def predict(net, loader):
    net.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(net(images.to(device)), dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


# 4. Training loop (keeps the best model, not just the last one)
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
best_score = -1.0
all_classes = list(range(len(class_names)))
print("\nStarting training...\n")

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    probs, labels = predict(model, val_loader)
    preds = probs.argmax(axis=1)
    accuracy = 100 * float((preds == labels).mean())
    balanced = 100 * float(balanced_accuracy_score(labels, preds))
    recalls = recall_score(labels, preds, labels=all_classes, average=None, zero_division=0)
    mel_recall = 100 * float(recalls[class_to_idx["mel"]])
    bcc_recall = 100 * float(recalls[class_to_idx["bcc"]])

    score = (balanced + mel_recall + bcc_recall) / 3
    saved = ""
    if epoch == EPOCHS - 1:  # always keep the final epoch
        best_score = score
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "disease_names": disease_names,
            "epoch": epoch + 1,
            "balanced_accuracy": balanced,
        }, MODEL_OUT)
        saved = "  <- final model saved"

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | loss {running_loss / len(train_loader):.3f} | "
        f"acc {accuracy:.1f}% | balanced {balanced:.1f}% | "
        f"melanoma recall {mel_recall:.1f}% | BCC recall {bcc_recall:.1f}% | "
        f"{time.time() - start:.0f}s{saved}"
    )

# 5. Final evaluation of the best model
print("\nEvaluating the best model...")
checkpoint = torch.load(MODEL_OUT, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
probs, labels = predict(model, val_loader)
preds = probs.argmax(axis=1)
confidence = probs.max(axis=1).astype(np.float64)
correct = preds == labels

accuracy = 100 * float(correct.mean())
balanced = 100 * float(balanced_accuracy_score(labels, preds))
cm = confusion_matrix(labels, preds, labels=all_classes)
report = classification_report(
    labels, preds, labels=all_classes, target_names=class_names,
    output_dict=True, zero_division=0,
)

per_class = []
for code in class_names:
    r = report[code]
    per_class.append({
        "code": code,
        "name": disease_names[code],
        "precision": round(100 * float(r["precision"]), 2),
        "recall": round(100 * float(r["recall"]), 2),
        "f1": round(100 * float(r["f1-score"]), 2),
        "support": int(r["support"]),
    })

calibration = []
ece = 0.0
edges = np.linspace(0, 1, 11)
for low, high in zip(edges[:-1], edges[1:]):
    mask = (confidence > low) & (confidence <= high)
    if mask.sum() == 0:
        continue
    bin_conf = 100 * float(confidence[mask].mean())
    bin_acc = 100 * float(correct[mask].mean())
    ece += abs(bin_acc - bin_conf) * int(mask.sum()) / len(labels)
    calibration.append({
        "bin": f"{int(round(low * 100))}-{int(round(high * 100))}%",
        "count": int(mask.sum()),
        "confidence": round(bin_conf, 2),
        "accuracy": round(bin_acc, 2),
    })

thresholds = []
for t in [0, 50, 60, 70, 80, 90]:
    mask = confidence >= t / 100
    thresholds.append({
        "threshold": t,
        "coverage": round(100 * float(mask.mean()), 2),
        "accuracy": round(100 * float(correct[mask].mean()), 2) if mask.sum() > 0 else None,
    })

results = {
    "class_codes": class_names,
    "class_names": [disease_names[c] for c in class_names],
    "accuracy": round(accuracy, 2),
    "balanced_accuracy": round(balanced, 2),
    "ece": round(float(ece), 2),
    "lesion_overlap": int(overlap),
    "n_train": int(len(train_df)),
    "n_val": int(len(val_df)),
    "per_class": per_class,
    "confusion_matrix": [[int(v) for v in row] for row in cm],
    "calibration": calibration,
    "thresholds": thresholds,
}
with open(EVAL_OUT, "w") as f:
    json.dump(results, f, indent=2)

by_code = {row["code"]: row["recall"] for row in per_class}
print("\n================ RESULTS ================")
print(f"Accuracy:           {results['accuracy']}%")
print(f"Balanced accuracy:  {results['balanced_accuracy']}%   (before: {BEFORE['balanced']}%)")
print(f"Melanoma recall:    {by_code['mel']}%   (before: {BEFORE['mel']}%)")
print(f"BCC recall:         {by_code['bcc']}%   (before: {BEFORE['bcc']}%)")
print(f"Calibration error:  {results['ece']} points")
print("\nPer-class recall:")
for row in per_class:
    print(f"  {row['name']:<24} {row['recall']:>6.1f}%  ({row['support']} images)")
print("\nSaved model:  ", MODEL_OUT)
print("Saved results:", EVAL_OUT)