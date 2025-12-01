import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. SYNTHETIC DATASET GENERATION
# -----------------------------------------------------

# Example features (you can replace these with real data):
# - actions_per_minute: Players with impossible APM may be cheating
# - accuracy: Players with superhuman accuracy or zero recoil
# - reaction_time: Very low reaction time might indicate aim-assist
# - movement_variation: Bots often move with low randomness
# - suspicious_reports: How many times player was reported

np.random.seed(42)

num_samples = 2000

# Legit players
legit = pd.DataFrame({
    "actions_per_minute": np.random.normal(120, 20, num_samples),
    "accuracy": np.random.normal(0.25, 0.08, num_samples),
    "reaction_time": np.random.normal(280, 40, num_samples),
    "movement_variation": np.random.normal(0.65, 0.1, num_samples),
    "suspicious_reports": np.random.poisson(0.3, num_samples),
    "cheater": np.zeros(num_samples)
})

# Cheaters
cheaters = pd.DataFrame({
    "actions_per_minute": np.random.normal(220, 35, num_samples),
    "accuracy": np.random.normal(0.75, 0.1, num_samples),
    "reaction_time": np.random.normal(90, 20, num_samples),
    "movement_variation": np.random.normal(0.20, 0.08, num_samples),
    "suspicious_reports": np.random.poisson(3.5, num_samples),
    "cheater": np.ones(num_samples)
})

# Combine datasets
df = pd.concat([legit, cheaters], ignore_index=True)

print("Dataset Loaded:")
print(df.head())

# -----------------------------------------------------
# 2. SPLIT DATA
# -----------------------------------------------------
X = df.drop("cheater", axis=1)
y = df["cheater"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------------------------------
# 3. TRAIN MODEL
# -----------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------------------
# 4. PREDICTIONS
# -----------------------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------------------
# 5. EVALUATIONS
# -----------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# 6. VISUALIZE CONFUSION MATRIX
# -----------------------------------------------------
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Legit", "Cheater"])
plt.yticks([0, 1], ["Legit", "Cheater"])

for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, val, ha='center', va='center', color='black')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 7. OUTPUT SAMPLE PREDICTIONS
# -----------------------------------------------------
sample_output = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})
print("\n=== SAMPLE PREDICTIONS (FIRST 20) ===")
print(sample_output)
