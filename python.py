from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = model.score(X_test_scaled, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# ===================== SINGLE PAGE DASHBOARD =====================

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")

# 2️⃣ Precision-Recall
precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
axes[0, 1].plot(recalls, precisions)
axes[0, 1].set_title("Precision-Recall Curve")

# 3️⃣ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
axes[0, 2].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
axes[0, 2].plot([0, 1], [0, 1], 'k--')
axes[0, 2].legend()
axes[0, 2].set_title("ROC Curve")

# 4️⃣ Feature Importance (Top 10)
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

axes[1, 0].barh(range(len(indices)), importances[indices])
axes[1, 0].set_yticks(range(len(indices)))
axes[1, 0].set_yticklabels(feature_names[indices])
axes[1, 0].set_title("Top 10 Feature Importance")

# 5️⃣–8️⃣ First 4 Boxplots
for i in range(4):
    row = 1 + (i // 3)
    col = (i % 3)
    sns.boxplot(x='target', y=feature_names[i], data=df, ax=axes[row, col])
    axes[row, col].set_title(f"{feature_names[i]}")

# Hide unused subplot
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()

# ================= USER INPUT =================

def predict_cell():
    print("\nEnter the following cell features:")
    input_features = []

    for i in range(5):
        while True:
            try:
                value = float(input(f"{feature_names[i]}: "))
                input_features.append(value)
                break
            except ValueError:
                print("Please enter a valid numeric value!")

    input_features += [0.0] * (30 - 5)

    input_scaled = scaler.transform([input_features])
    pred = model.predict(input_scaled)[0]

    if pred == 0:
        print("\nResult: Cancerous (Malignant) cell detected")
    else:
        print("\nResult: Not Cancerous (Benign) cell detected")


predict_cell()
