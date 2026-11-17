# ==============================================================================
# Hands-on Lab Evaluation - 3 (Module III)
# Topic: Classification Models on Breast Cancer Wisconsin Dataset
# Student Implementation: 5 Supervised Classification Algorithms
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Plot settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

os.makedirs("visualizations", exist_ok=True)

# ------------------------------------------------------------------------------
# Task 1: Load the Dataset and Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
print("=" * 60)
print("TASK 1: Loading and Exploring Dataset")
print("=" * 60)

# Load dataset directly from GitHub repository
github_url = "https://raw.githubusercontent.com/ayankhan96686-create/ml-hand-on-lab3/main/archive/data.csv"
try:
    df = pd.read_csv(github_url)
    print("Dataset loaded successfully from GitHub!")
except Exception:
    local_path = "../archive/data.csv" if os.path.exists("../archive/data.csv") else "archive/data.csv"
    df = pd.read_csv(local_path)
    print(f"Loaded from local file: {local_path}")
print(f"Dataset Loaded. Total Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 1. Display first 10 records
print("\n--- First 10 Records ---")
print(df.head(10))

# 2. Display last 10 records
print("\n--- Last 10 Records ---")
print(df.tail(10))

# 3. Dataset Info
print("\n--- Dataset Info ---")
df.info()

# 4. Summary Statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# 5. Check for missing values
print("\n--- Missing Values Count ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 6. Target variable distribution (Benign vs Malignant)
print("\n--- Diagnosis Count (B = Benign, M = Malignant) ---")
print(df['diagnosis'].value_counts())

# Plot Histograms of important features
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(features):
    ax = axes[i // 2, i % 2]
    sns.histplot(data=df, x=col, hue='diagnosis', kde=True, ax=ax, palette={'B': 'green', 'M': 'red'})
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("visualizations/01_feature_distributions_histograms.png", dpi=300)
plt.close()

# Plot Boxplots of important features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(features):
    ax = axes[i // 2, i % 2]
    sns.boxplot(data=df, x='diagnosis', y=col, ax=ax, palette={'B': 'green', 'M': 'red'}, hue='diagnosis', legend=False)
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig("visualizations/02_feature_distributions_boxplots.png", dpi=300)
plt.close()

print("Histograms and boxplots saved in 'visualizations/' folder.")

# ------------------------------------------------------------------------------
# Task 2: Data Preprocessing
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2: Data Preprocessing")
print("=" * 60)

# Drop useless columns ('id' is just patient ID, 'Unnamed: 32' is an empty column)
drop_cols = [c for c in ['id', 'Unnamed: 32'] if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")

# Encode target column: 'M' (Malignant) -> 1, 'B' (Benign) -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("Encoded 'diagnosis': M -> 1, B -> 0")

# Save cleaned dataset to CSV
df.to_csv("cleaned_breast_cancer.csv", index=False)
print("Saved cleaned data as 'cleaned_breast_cancer.csv'.")

# Separate features (X) and label (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split into 70% Train and 30% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler (mean=0, variance=1).")

# ------------------------------------------------------------------------------
# Task 3: Logistic Regression & Decision Tree Classifier
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3: Logistic Regression and Decision Tree")
print("=" * 60)

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("--- Logistic Regression ---")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_lr), 4))
print("Precision:", round(precision_score(y_test, y_pred_lr), 4))
print("Recall:   ", round(recall_score(y_test, y_pred_lr), 4))
print("F1-Score: ", round(f1_score(y_test, y_pred_lr), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_prob_lr), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# 2. Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

print("\n--- Decision Tree ---")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_dt), 4))
print("Precision:", round(precision_score(y_test, y_pred_dt), 4))
print("Recall:   ", round(recall_score(y_test, y_pred_dt), 4))
print("F1-Score: ", round(f1_score(y_test, y_pred_dt), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_prob_dt), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# ------------------------------------------------------------------------------
# Task 4: Random Forest Classifier & Support Vector Machine (SVM)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4: Random Forest and Support Vector Machine (SVM)")
print("=" * 60)

# 1. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("--- Random Forest ---")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_rf), 4))
print("Precision:", round(precision_score(y_test, y_pred_rf), 4))
print("Recall:   ", round(recall_score(y_test, y_pred_rf), 4))
print("F1-Score: ", round(f1_score(y_test, y_pred_rf), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_prob_rf), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 2. Support Vector Machine (SVM)
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_score_svm = svm.decision_function(X_test_scaled)

print("\n--- Support Vector Machine (SVM) ---")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_svm), 4))
print("Precision:", round(precision_score(y_test, y_pred_svm), 4))
print("Recall:   ", round(recall_score(y_test, y_pred_svm), 4))
print("F1-Score: ", round(f1_score(y_test, y_pred_svm), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_score_svm), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# ------------------------------------------------------------------------------
# Task 5: Gaussian Naive Bayes Classifier & Final 5-Model Comparison
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5: Naive Bayes and Comparison of All 5 Models")
print("=" * 60)

# 1. Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
y_prob_nb = nb.predict_proba(X_test_scaled)[:, 1]

print("--- Naive Bayes ---")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_nb), 4))
print("Precision:", round(precision_score(y_test, y_pred_nb), 4))
print("Recall:   ", round(recall_score(y_test, y_pred_nb), 4))
print("F1-Score: ", round(f1_score(y_test, y_pred_nb), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_prob_nb), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

# 2. Comparison Table
models_data = [
    {
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr),
        'Recall': recall_score(y_test, y_pred_lr),
        'F1-Score': f1_score(y_test, y_pred_lr),
        'ROC-AUC': roc_auc_score(y_test, y_prob_lr)
    },
    {
        'Model': 'Decision Tree',
        'Accuracy': accuracy_score(y_test, y_pred_dt),
        'Precision': precision_score(y_test, y_pred_dt),
        'Recall': recall_score(y_test, y_pred_dt),
        'F1-Score': f1_score(y_test, y_pred_dt),
        'ROC-AUC': roc_auc_score(y_test, y_prob_dt)
    },
    {
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf),
        'Recall': recall_score(y_test, y_pred_rf),
        'F1-Score': f1_score(y_test, y_pred_rf),
        'ROC-AUC': roc_auc_score(y_test, y_prob_rf)
    },
    {
        'Model': 'Support Vector Machine (SVM)',
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Precision': precision_score(y_test, y_pred_svm),
        'Recall': recall_score(y_test, y_pred_svm),
        'F1-Score': f1_score(y_test, y_pred_svm),
        'ROC-AUC': roc_auc_score(y_test, y_score_svm)
    },
    {
        'Model': 'Naïve Bayes',
        'Accuracy': accuracy_score(y_test, y_pred_nb),
        'Precision': precision_score(y_test, y_pred_nb),
        'Recall': recall_score(y_test, y_pred_nb),
        'F1-Score': f1_score(y_test, y_pred_nb),
        'ROC-AUC': roc_auc_score(y_test, y_prob_nb)
    }
]

comparison_df = pd.DataFrame(models_data).set_index('Model')
print("\n" + "=" * 60)
print("FINAL BENCHMARK COMPARISON TABLE")
print("=" * 60)
print(comparison_df.round(4))

# 3. Plot Confusion Matrices for all 5 models side by side
predictions_map = {
    'Logistic Regression': y_pred_lr,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svm,
    'Naïve Bayes': y_pred_nb
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (name, pred) in enumerate(predictions_map.items()):
    ax = axes[i]
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f"{name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_yticklabels(['Benign', 'Malignant'])

fig.delaxes(axes[5])  # delete extra subplot
plt.suptitle("Confusion Matrices of All 5 Models", fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig("visualizations/05_confusion_matrices.png", dpi=300)
plt.close()

# 4. Plot ROC Curves
curves_map = {
    'Logistic Regression': y_prob_lr,
    'Decision Tree': y_prob_dt,
    'Random Forest': y_prob_rf,
    'SVM': y_score_svm,
    'Naïve Bayes': y_prob_nb
}

plt.figure(figsize=(9, 7))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for (name, scores), color in zip(curves_map.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_val = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (AUC = 0.50)')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves Comparison of All Classifiers", fontsize=14, weight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("visualizations/06_roc_curves.png", dpi=300)
plt.close()

# 5. Bar Chart comparison
comparison_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')
plt.title("Performance Metric Comparison Across All 5 Classifiers", fontsize=14, weight='bold')
plt.ylabel("Score")
plt.ylim(0.70, 1.02)
plt.xticks(rotation=20, ha='right')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("visualizations/07_models_performance_comparison.png", dpi=300)
plt.close()

print("\nPlots saved successfully in 'visualizations/' folder.")

# 6. Conclusion (150-200 words)
conclusion = """
In this clinical machine learning investigation, five supervised classification algorithms were trained and rigorously evaluated on the Breast Cancer Wisconsin Diagnostic dataset: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Gaussian Naive Bayes. Diagnostic features extracted from digitized fine needle aspirates (FNA) were normalized using StandardScaler to eliminate dimensional scale discrepancies.

Logistic Regression demonstrated outstanding diagnostic capability, attaining the highest Accuracy (97.08%), Recall (93.75%), and F1-Score (96.00%) with an exemplary ROC-AUC of 0.9975. In oncology applications, Recall is the paramount clinical metric because a False Negative (misclassifying a malignant tumor as benign) can delay life-saving medical interventions. Logistic Regression produced only 4 False Negatives out of 64 malignant cases, outperforming Random Forest (6 FN) and SVM (7 FN). Furthermore, Random Forest achieved perfect Precision (100%) and 96.49% Accuracy, demonstrating immense utility when zero false alarms are desired. Decision Tree yielded the lowest generalization (90.06% Accuracy, 10 FN), suffering from local greedy variance. Consequently, Logistic Regression is recommended as the premier classification model for clinical deployment, paired with Random Forest for ensemble verification.
"""

print("\n" + "=" * 60)
print("EXECUTIVE CONCLUSION (150-200 WORDS):")
print("=" * 60)
print(conclusion.strip())
