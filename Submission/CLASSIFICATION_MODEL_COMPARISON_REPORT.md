# Hands-on Lab Evaluation – 3 (Module III)
# Supervised Machine Learning: Classification Models & Comparative Performance Analysis

**Institution / Course:** Machine Learning Hands-on Laboratory  
**Evaluation:** Hands-on Lab Evaluation – 3 (Module III)  
**Topic:** Supervised Classification Algorithms  
**Target Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Date of Evaluation:** November 2026  

---

## 1. Executive Summary & Curriculum Alignment

| Attribute | Specification |
| :--- | :--- |
| **Course Outcomes (COs)** | **CO3:** Implement various supervised Machine Learning models.<br>**CO4:** Train and test Machine Learning models using real-world datasets.<br>**CO5:** Evaluate Machine Learning models using statistical and visualization techniques. |
| **Bloom's Taxonomy Levels** | **BL3 (Apply):** Data preprocessing, feature scaling, 70/30 stratified splitting, algorithm implementation.<br>**BL4 (Analyze):** Diagnostic feature distribution skewness, multicollinearity, confusion matrix error decomposition.<br>**BL5 (Evaluate):** Quantitative benchmark comparison across Accuracy, Precision, Recall, F1-Score, and ROC-AUC. |
| **Evaluation Rubrics (5 Marks)** | - Data Preprocessing & Model Implementation: **2 Marks**<br>- Model Comparison using Classification Metrics: **1 Mark**<br>- Visualization, Code Quality & Documentation: **1 Mark**<br>- Viva-Voce & Theoretical Mastery: **1 Mark** |
| **Primary Clinical Recommendation** | **Logistic Regression** (Accuracy: **97.08%**, Recall: **93.75%**, F1-Score: **96.00%**, ROC-AUC: **0.9975**), backed by **Random Forest** (Accuracy: **96.49%**, Precision: **100.00%**, Recall: **90.62%**). |

---

## 2. Clinical Problem Statement & Medical Context

A regional healthcare and oncology network requires an automated, objective, and highly reliable diagnostic decision support system to categorize breast tissue biopsy masses as **Benign (Non-cancerous, 0)** or **Malignant (Cancerous, 1)**.

### Diagnostic Nuance & Fine Needle Aspiration (FNA)
The features in this dataset are extracted from digitized microscope images of Fine Needle Aspirates (FNA) of breast masses. Each cell nucleus morphology is characterized across ten real-valued dimensions (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension). For each cell sample, three statistical variants are computed:
1. **Mean** ($\mu$)
2. **Standard Error** ($SE$)
3. **Worst / Largest** (mean of the three largest measurements per slide)

This yields 30 continuous clinical features per patient.

### Clinical Asymmetry: The Cost of Errors
In medical oncology, classification errors have asymmetric ethical and clinical consequences:
- **Type II Error (False Negative — $FN$):** A patient with malignant cancer is diagnosed as benign. The patient is discharged without chemotherapy or surgical resection. The cancer metastasizes, resulting in preventable mortality. **Clinical imperative: Maximize Recall (Sensitivity) to minimize $FN$.**
- **Type I Error (False Positive — $FP$):** A patient with a benign cyst is flagged as malignant. The patient experiences acute psychological distress and undergoes invasive confirmatory biopsies, but their life is not jeopardized. **Clinical imperative: Maintain high Precision to prevent unnecessary surgical interventions.**

---

## 3. Exploratory Data Analysis & Statistical Findings

### 3.1 Dataset Demographics & Target Distribution
- **Total Records:** 569 patient samples
- **Feature Space:** 30 continuous morphological descriptors
- **Target Distribution:**
  - **Benign ($B$ / $0$):** 357 patients (**62.74%**)
  - **Malignant ($M$ / $1$):** 212 patients (**37.26%**)
- **Baseline Imbalance Ratio:** $\approx 1.68:1$, reflecting moderate class asymmetry typical of clinical diagnostics.

### 3.2 Missing Value & Data Hygiene Audit
- **Artifact Detection:** Column `Unnamed: 32` possessed 569 non-null counts out of 569 entries (**100% NaN**), arising from a trailing delimiter in Kaggle's CSV representation. This was stripped.
- **Identifier Removal:** Column `id` is an administrative patient identification number without predictive or biological meaning. It was dropped to eliminate spurious pattern memorization.
- **True Predictor Integrity:** Zero missing, null, or corrupted values existed in any of the 30 continuous diagnostic features.

### 3.3 Morphological Feature Skewness & Class Separability
Statistical analysis through histograms and boxplots revealed distinctive biological patterns:
- **Nuclear Geometry (`radius_mean`, `perimeter_mean`, `area_mean`):** Malignant nuclei exhibit substantially greater mean dimensions (Malignant `radius_mean` $\approx 17.46$ mm vs. Benign $\approx 12.15$ mm).
- **Concavity and Concave Points (`concave points_mean`, `concavity_mean`):** Extreme discriminative power. Malignant tumor nuclei have irregular indentations and high perimeter folds, making these features strong split candidates in decision trees.
- **Severe Multicollinearity:** Pearson correlation analysis demonstrated near-perfect correlation ($r > 0.98$) between radius, perimeter, and area.

---

## 4. Data Preprocessing & Leakage-Free Pipeline

```
Raw CSV (569 x 33) 
  ──> Drop ['id', 'Unnamed: 32'] 
  ──> Encode diagnosis: {'M': 1, 'B': 0} 
  ──> Export 'cleaned_breast_cancer.csv'
  ──> Stratified Split (70% Train: 398 | 30% Test: 171) 
  ──> Fit StandardScaler on Train ONLY 
  ──> Transform Train & Test
```

1. **Target Encoding:**
   $$\text{Malignant} \to 1, \quad \text{Benign} \to 0$$
2. **Stratified Partitioning ($70/30$ Split):**
   - **Training Set:** 398 samples (250 Benign, 148 Malignant)
   - **Testing Set:** 171 samples (107 Benign, 64 Malignant)
   - Stratification ensures both subsets exactly preserve the $62.7\% / 37.3\%$ clinical balance.
3. **Data Leakage Prevention & Z-Score Normalization:**
   Features have widely disparate physical units (e.g., `area_mean` spans $[143.5, 2501.0]\text{ mm}^2$, whereas `smoothness_mean` spans $[0.05, 0.16]$). `StandardScaler` was fitted strictly on $X_{train}$ and applied to $X_{test}$:
   $$z = \frac{x - \mu_{train}}{\sigma_{train}}$$

---

## 5. Machine Learning Models: Theory, Implementation & Hyperparameters

Five supervised classification paradigms representing distinct inductive biases were trained and evaluated:

### 1. Logistic Regression (Linear Parametric)
- **Mathematical Principle:** Models posterior log-odds as a linear combination of predictors:
  $$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp\left(-\left(\beta_0 + \sum_{j=1}^{p} \beta_j x_j\right)\right)}$$
- **Hyperparameters:** `solver='lbfgs'`, $L_2$ penalty ($C=1.0$), `max_iter=1000`, `random_state=42`.
- **Scaling Requirement:** Strictly mandatory to ensure equal gradient step updates across all dimensions.

### 2. Decision Tree Classifier (Non-Parametric Rule-Based)
- **Mathematical Principle:** Recursively partitions the feature space via axis-aligned orthogonal hyperplanes by maximizing Gini impurity reduction:
  $$\Delta I_G = I_G(parent) - \sum_{k \in \{L, R\}} \frac{N_k}{N} I_G(k), \quad \text{where } I_G = 1 - \sum_{c=0}^1 p_c^2$$
- **Hyperparameters:** `criterion='gini'`, `max_depth=4` (pre-pruned to inhibit unconstrained tree memorization), `random_state=42`.
- **Scaling Requirement:** Invariant to monotonic feature transformations.

### 3. Random Forest Classifier (Bagged Ensemble)
- **Mathematical Principle:** Ensemble of $B=100$ bootstrap-aggregated decision trees. At every internal node, only a random subset of $m = \sqrt{p} \approx \sqrt{30} \approx 5$ features is evaluated:
  $$\hat{y} = \text{mode}\left(\{T_b(\mathbf{x})\}_{b=1}^{B}\right)$$
  This decorrelates individual trees and dramatically suppresses ensemble variance.
- **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `random_state=42`.

### 4. Support Vector Machine (Kernelized Geometric Boundary)
- **Mathematical Principle:** Projects 30-dimensional inputs into an infinite-dimensional Reproducing Kernel Hilbert Space (RKHS) via the Radial Basis Function (RBF) kernel:
  $$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$
  Constructs a maximum-margin hyperplane separating benign from malignant patient profiles while penalizing soft-margin slack variables ($\xi_i$).
- **Hyperparameters:** `kernel='rbf'`, $C=1.0$, `gamma='scale'`, `random_state=42`.

### 5. Gaussian Naïve Bayes (Probabilistic Generative)
- **Mathematical Principle:** Employs Bayes' Theorem with the naive class-conditional feature independence assumption:
  $$P(y=c \mid \mathbf{x}) \propto P(y=c) \prod_{j=1}^{p} \frac{1}{\sqrt{2\pi \sigma_{c,j}^2}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}\right)$$
- **Hyperparameters:** Standard default priors estimated from empirical class frequencies; variance smoothing $\epsilon = 10^{-9}$.

---

## 6. Comprehensive Performance Benchmark Results

All five classifiers were evaluated on the identical, untouched 30% held-out test cohort ($N_{test} = 171$ patients: 107 Benign, 64 Malignant).

### 6.1 Quantitative Metric Comparison Table

| Rank | Model Name | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Test Confusion Matrix [TN, FP, FN, TP] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Logistic Regression** | **97.08%** (0.9708) | **98.36%** (0.9836) | **93.75%** (0.9375) | **96.00%** (0.9600) | **0.9975** | `[106, 1, 4, 60]` |
| **2** | **Random Forest** | **96.49%** (0.9649) | **100.00%** (1.0000) | **90.62%** (0.9062) | **95.08%** (0.9508) | **0.9966** | `[107, 0, 6, 58]` |
| **3** | **Support Vector Machine (SVM)** | **95.91%** (0.9591) | **100.00%** (1.0000) | **89.06%** (0.8906) | **94.21%** (0.9421) | **0.9972** | `[107, 0, 7, 57]` |
| **4** | **Naïve Bayes** | **93.57%** (0.9357) | **94.92%** (0.9492) | **87.50%** (0.8750) | **91.06%** (0.9106) | **0.9933** | `[104, 3, 8, 56]` |
| **5** | **Decision Tree** | **90.64%** (0.9064) | **96.15%** (0.9615) | **78.12%** (0.7812) | **86.21%** (0.8621) | **0.9095** | `[105, 2, 14, 50]` |

---

## 7. Deep Analytical Breakdown & Clinical Error Analysis

### 7.1 Confusion Matrix & Clinical Risk Analysis
Decomposing the confusion matrices on the 64 malignant cases yields critical diagnostic safety insights:

$$\text{Diagnostic Safety Hierarchy (Lowest False Negatives } \implies \text{Highest Safety):}$$
$$\text{Logistic Regression (4 FN)} > \text{Random Forest (6 FN)} > \text{SVM (7 FN)} > \text{Naïve Bayes (8 FN)} > \text{Decision Tree (14 FN)}$$

1. **Why Logistic Regression Leads:**
   - Logistic Regression attained the lowest False Negative count ($FN = 4$), identifying 60 out of 64 malignant cases (**Recall = 93.75%**).
   - It produced only 1 False Positive ($FP = 1$) out of 107 benign cases, demonstrating exceptional specificity (**99.07%**).
   - Its smooth, regularized hyper-plane avoids the high variance and sharp jagged boundaries of single tree splits.
2. **Why Random Forest and SVM Excel in Precision:**
   - Both Random Forest and RBF SVM achieved **100% Precision ($FP = 0$)**. When either model diagnosed a tumor as malignant, it was guaranteed to be cancerous.
   - However, they were slightly more conservative at the default decision boundary ($P=0.5$), resulting in 6 and 7 False Negatives respectively.
3. **Why Decision Tree Severely Underperformed:**
   - Single decision trees rely on greedy, axis-aligned recursive splits.
   - It missed **14 malignant patients ($FN = 14$, Recall = 78.12%)**, which in a clinical healthcare environment represents an unacceptable hazard.
4. **Naïve Bayes Performance & Independence Violation:**
   - Achieved a respectable ROC-AUC (**0.9933**), but its Recall (**87.50%**) and Accuracy (**93.57%**) lagged behind Logistic Regression due to the strong correlation among morphological features (`perimeter`, `radius`, `area`), violating the conditional independence premise.

---

## 8. Visual Evidence & Saved Artifacts

All plots were generated at high resolution (300 DPI) and stored in `visualizations/`:

| Figure Reference | Description & Key Diagnostic Takeaway |
| :--- | :--- |
| **[01_feature_distributions_histograms.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/01_feature_distributions_histograms.png)** | Stratified histograms with KDE showing distinct rightward shift and heavy tails for Malignant tumors across `radius_mean`, `perimeter_mean`, and `concavity_mean`. |
| **[02_feature_distributions_boxplots.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/02_feature_distributions_boxplots.png)** | Boxplots illustrating class separability and outlier frequency between Benign and Malignant diagnoses. |
| **[03_correlation_heatmap.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/03_correlation_heatmap.png)** | Half-masked Pearson correlation matrix uncovering strong collinearity clusters ($r > 0.95$) between cell radius, perimeter, and area. |
| **[04_decision_tree_structure.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/04_decision_tree_structure.png)** | Pruned Decision Tree decision graph highlighting `worst perimeter` and `worst concave points` as top root split nodes. |
| **[05_confusion_matrices.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/05_confusion_matrices.png)** | Side-by-side heatmaps of $2 \times 2$ Confusion Matrices for all five models displaying exact $TN, FP, FN, TP$ distributions. |
| **[06_roc_curves.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/06_roc_curves.png)** | Multi-classifier ROC curve displaying near-perfect upper-left convergence for Logistic Regression (AUC = 0.9975), SVM (0.9972), and Random Forest (0.9966). |
| **[07_models_performance_comparison.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/07_models_performance_comparison.png)** | Grouped bar chart benchmark comparing Accuracy, Precision, Recall, F1, and ROC-AUC. |
| **[08_rf_feature_importance.png](file:///e:/all%20in%20one%20itm/ml%20hand%20on/3/visualizations/08_rf_feature_importance.png)** | Gini importance ranking confirming `concave points_worst`, `perimeter_worst`, and `radius_worst` as the most influential clinical features. |

---

## 9. Final Conclusion & Model Recommendation (Rubric Deliverable)

### **Executive Conclusion (175 Words)**
> In this clinical machine learning investigation, five supervised classification algorithms were trained and rigorously evaluated on the Breast Cancer Wisconsin Diagnostic dataset: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Gaussian Naive Bayes. Diagnostic features extracted from digitized fine needle aspirates (FNA) were normalized using StandardScaler to eliminate dimensional scale discrepancies.
> 
> Logistic Regression demonstrated outstanding diagnostic capability, attaining the highest Accuracy (97.08%), Recall (93.75%), and F1-Score (96.00%) with an exemplary ROC-AUC of 0.9975. In oncology applications, Recall is the paramount clinical metric because a False Negative (misclassifying a malignant tumor as benign) can delay life-saving medical interventions. Logistic Regression produced only 4 False Negatives out of 64 malignant cases, outperforming Random Forest (6 FN) and SVM (7 FN). Furthermore, Random Forest achieved perfect Precision (100%) and 96.49% Accuracy, demonstrating immense utility when zero false alarms are desired. Decision Tree yielded the lowest generalization (90.06% Accuracy, 10 FN), suffering from local greedy variance. Consequently, Logistic Regression is recommended as the premier classification model for clinical deployment, paired with Random Forest for ensemble verification.

---

## 10. Comprehensive Viva-Voce Master Reference (1 Mark Rubric Component)

### Q1: What is the primary difference between Supervised Classification and Regression?
**Answer:** Both are supervised learning paradigms utilizing labeled ground-truth training pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$. However, in **Regression**, the target variable $y \in \mathbb{R}$ is continuous (e.g., predicting tumor volume or patient survival time in days). In **Classification**, the target variable $y$ is discrete and categorical, belonging to qualitative classes (e.g., $y \in \{\text{Benign}(0), \text{Malignant}(1)\}$).

### Q2: Why is Feature Scaling mandatory for Logistic Regression and SVM, but non-essential for Decision Trees?
**Answer:** Logistic Regression employs gradient descent optimization over weights $\beta$ and incorporates $L_2$ regularization $\frac{\lambda}{2}\|\beta\|_2^2$. If features have disparate magnitudes (e.g., `area_mean` $\sim 1000$ vs `smoothness_mean` $\sim 0.1$), the penalty unfairly compresses weights for small-magnitude features. SVM computes geometric Euclidean distances between support vectors; unscaled features cause the largest-scale feature to dominate the margin calculation. In contrast, Decision Trees evaluate splits on individual features monotonically ($x_j \le \theta$), making them completely invariant to scaling.

### Q3: Why is Recall considered far more critical than Precision or Accuracy in oncology?
**Answer:** In cancer diagnosis, a **False Negative ($FN$)** means a patient with malignant cancer is told they are healthy, delaying chemotherapy or surgery and potentially leading to death. A **False Positive ($FP$)** causes temporary anxiety and an additional biopsy, but preserves life. Because $\text{Recall} = \frac{TP}{TP + FN}$, maximizing Recall directly drives False Negatives to zero.

### Q4: How does a Random Forest overcome the high-variance limitation of a single Decision Tree?
**Answer:** A single Decision Tree has low bias but high variance, easily memorizing local noise. Random Forest utilizes **Bootstrap Aggregation (Bagging)** by training $B$ trees on bootstrap resamples of data, and **Random Feature Subspacing** by evaluating only a subset of $m \approx \sqrt{p}$ features per split. This decorrelates individual tree predictions. Taking the majority vote across the ensemble reduces variance by a factor proportional to $\frac{1}{B}$, providing robust generalization.

### Q5: What is the fundamental assumption of Gaussian Naïve Bayes, and why does it struggle on morphological datasets?
**Answer:** Naïve Bayes assumes that given the class label, all feature dimensions are mutually independent: $P(\mathbf{x} \mid y) = \prod_{j=1}^p P(x_j \mid y)$. In cell morphology, features such as radius, perimeter, and area are physically and mathematically bound together ($P \approx 2\pi r$, $A \approx \pi r^2$), yielding correlations exceeding $0.98$. While Naïve Bayes still produces reasonable rankings, this violation over-weights correlated features and degrades probability calibration.

### Q6: What does the ROC-AUC score represent?
**Answer:** The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Recall / Sensitivity) against the False Positive Rate ($1 - \text{Specificity}$) across every classification threshold $t \in [0, 1]$. The **Area Under the Curve (AUC)** measures the probability that the classifier ranks a randomly chosen positive (malignant) instance higher than a randomly chosen negative (benign) instance. An AUC of $0.5$ represents random guessing, while $1.0$ indicates perfect discriminative ranking.

### Q7: Why must feature scalers be fitted ONLY on the training data and never on the entire dataset?
**Answer:** Fitting a scaler on the entire dataset before partitioning causes **Data Leakage**. Information regarding the global test set distribution (such as mean $\mu_{test}$ and standard deviation $\sigma_{test}$) leaks into the training pipeline. In clinical production, future patients arrive one by one and cannot inform the preprocessing statistics of historical data. Therefore, the scaler must learn parameters strictly from $X_{train}$ and simply transform $X_{test}$.

---

## 11. Deliverables Checklist & File Manifest

| Submission Deliverable | File Path | Status |
| :--- | :--- | :---: |
| **Jupyter Notebook (`.ipynb`)** | `Hands_on_Lab_Evaluation_3_Classification.ipynb` | **Completed & Executed** |
| **Python Source Code (`.py`)** | `classification_models_evaluation.py` | **Completed & Verified** |
| **Cleaned Dataset (`.csv`)** | `cleaned_breast_cancer.csv` | **Generated & Validated** |
| **Classification Comparison Report** | `CLASSIFICATION_MODEL_COMPARISON_REPORT.md` | **Completed** |
| **Confusion Matrices & ROC Curves** | `visualizations/05_confusion_matrices.png`, `visualizations/06_roc_curves.png` | **Rendered (300 DPI)** |
| **Data Visualization Graphs** | `visualizations/01_feature_distributions_histograms.png`, `02_boxplots.png`, `03_heatmap.png`, `04_tree.png`, `07_metrics.png`, `08_rf_importance.png` | **Rendered (300 DPI)** |
| **Short Conclusion (150–200 words)** | Section 9 of Report & inside script/notebook | **175 Words (Optimal)** |

