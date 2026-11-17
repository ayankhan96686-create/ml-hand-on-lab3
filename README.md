# Hands-on Lab Evaluation – 3 (Module III)
## Supervised Classification Models: Breast Cancer Wisconsin Diagnostic

This repository contains the complete implementation and comparative evaluation of **5 Supervised Machine Learning Classification Algorithms** on the **Breast Cancer Wisconsin Dataset**.

---

### Project Overview
The objective is to classify breast tumors as either **Benign (0)** or **Malignant (1)** using 30 morphological features extracted from Fine Needle Aspirates (FNA).

The 5 classification models implemented and evaluated are:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)**
5. **Gaussian Naïve Bayes**

---

### Dataset Details
- **Source:** [Kaggle Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) / UCI Machine Learning Repository
- **Total Records:** 569 patients
- **Classes:** 357 Benign (62.7%), 212 Malignant (37.3%)
- **Features:** 30 numerical diagnostic measurements (radius, texture, perimeter, area, smoothness, etc.)

---

### Performance Comparison

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **97.08%** | **98.36%** | **93.75%** *(Best)* | **96.00%** | **0.9975** |
| **Random Forest** | **96.49%** | **100.00%** | **90.62%** | **95.08%** | **0.9966** |
| **Support Vector Machine (SVM)** | **95.91%** | **100.00%** | **89.06%** | **94.21%** | **0.9972** |
| **Naïve Bayes** | **93.57%** | **94.92%** | **87.50%** | **91.06%** | **0.9933** |
| **Decision Tree** | **90.64%** | **96.15%** | **78.12%** | **86.21%** | **0.9095** |

*Note: In cancer detection, **Recall** is the most critical metric because a False Negative (missing a malignant tumor) can be fatal. Logistic Regression had only 4 False Negatives, making it the most reliable model.*

---

### Key Visualizations
Generated plots are stored in the `visualizations/` folder:
- `01_feature_distributions_histograms.png`: Histograms of key features (radius, texture, perimeter, area)
- `02_feature_distributions_boxplots.png`: Boxplots showing outliers and class spread
- `05_confusion_matrices.png`: Confusion matrices for all 5 models side by side
- `06_roc_curves.png`: Combined ROC curves showing AUC values
- `07_models_performance_comparison.png`: Grouped bar chart comparing all metrics across models

---

### Repository Structure
```
.
├── Hands_on_Lab_Evaluation_3_Classification.ipynb  # Executed Jupyter Notebook
├── classification_models_evaluation.py             # Complete Python script
├── cleaned_breast_cancer.csv                       # Cleaned dataset (ready to use)
├── CLASSIFICATION_MODEL_COMPARISON_REPORT.md       # Comprehensive evaluation report
├── visualizations/                                 # Generated evaluation charts
│   ├── 01_feature_distributions_histograms.png
│   ├── 02_feature_distributions_boxplots.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_decision_tree_structure.png
│   ├── 05_confusion_matrices.png
│   ├── 06_roc_curves.png
│   └── 07_models_performance_comparison.png
├── archive/
│   └── data.csv                                    # Original raw dataset
└── README.md
```

---

### How to Run

1. **Install requirements:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. **Run Python script:**
   ```bash
   python classification_models_evaluation.py
   ```

3. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook Hands_on_Lab_Evaluation_3_Classification.ipynb
   ```
