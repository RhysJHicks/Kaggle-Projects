import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve, StratifiedKFold

# ==============================
# Load dataset
# ==============================
df = pd.read_excel(r"../Rice_Cammeo_Osmancik.xlsx")

features = [
    'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
    'Eccentricity', 'Convex_Area', 'Extent'
]

X = df[features]
y = df['Class']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(random_state=42)

# =========================================================
# 1. RESPONSE SPACE: n_estimators
# =========================================================
param_range_estimators = [5, 10, 15, 20, 25]

train_scores, test_scores = validation_curve(
    rf,
    X, y,
    param_name="n_estimators",
    param_range=param_range_estimators,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# =========================================================
# 2. RESPONSE SPACE: max_depth
# =========================================================
param_range_depth = [1, 2, 3, 4, 5]

train_scores_d, test_scores_d = validation_curve(
    rf,
    X, y,
    param_name="max_depth",
    param_range=param_range_depth,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

train_mean_d = np.mean(train_scores_d, axis=1)
test_mean_d = np.mean(test_scores_d, axis=1)

# =========================================================
# PLOTTING (2 lines each: Train vs Test)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -----------------------------
# n_estimators plot
# -----------------------------
axes[0].plot(param_range_estimators, train_mean, marker='o', label="Train Accuracy")
axes[0].plot(param_range_estimators, test_mean, marker='o', label="Test Accuracy")
axes[0].set_title("Response Space: n_estimators")
axes[0].set_xlabel("Number of Trees")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# -----------------------------
# max_depth plot
# -----------------------------
axes[1].plot([str(x) for x in param_range_depth], train_mean_d, marker='o', label="Train Accuracy")
axes[1].plot([str(x) for x in param_range_depth], test_mean_d, marker='o', label="Test Accuracy")
axes[1].set_title("Response Space: max_depth")
axes[1].set_xlabel("Tree Depth")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()