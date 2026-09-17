# 💳 Credit Wise — Loan Approval Prediction System

A complete Machine Learning classification project that predicts whether a loan application will be **approved or rejected** based on applicant financial profile, demographics, and credit history. Built using Python and Scikit-learn.

---

## 📌 Problem Statement

Banks and financial institutions process thousands of loan applications daily. Manual evaluation is slow and prone to bias. This project builds an **automated loan approval system** using supervised ML classification algorithms to predict loan eligibility based on applicant data.

---

## 🎯 Objective

> **Binary Classification Task**: Predict Loan Approval → `1 (Approved)` or `0 (Rejected)`

---

## 📂 Dataset Features

| Feature | Description |
|---|---|
| `Applicant_Income` | Monthly income of the primary applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Age` | Age of the applicant |
| `Dependents` | Number of dependents |
| `Existing_Loans` | Number of existing loans |
| `Savings` | Applicant's savings amount |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan repayment term (months) |
| `Education_Level` | Education level (0/1) |
| `Property_Area` | Rural / Semiurban / Urban |
| `Gender` | Male / Female |
| `Employer_Category` | Government / MNC / Private / Unemployed |
| `DTI_Ratio_sq` | Debt-to-Income ratio squared (engineered feature) |
| `Credit_Score_sq` | Credit score squared (engineered feature) |
| `Applicant_Income_log` | Log-transformed income (engineered feature) |

---

## 🔧 Feature Engineering

- **Log Transformation** on `Applicant_Income` to reduce skewness
- **Squared Terms** of `DTI_Ratio` and `Credit_Score` to capture non-linear relationships
- **One-Hot Encoding** for categorical columns (`Property_Area`, `Gender`, `Employer_Category`)
- **Missing Value Imputation** using mean/median strategy
- **StandardScaler** applied to normalize numerical features before model training

---

## 🤖 ML Models Used

### 1. Logistic Regression
> Binary classification using linear decision boundary with sigmoid activation.

| Metric | Score |
|---|---|
| ✅ **Accuracy** | **88.0%** |
| 🎯 Precision | 78.46% |
| 📈 Recall | 83.61% |
| 🏆 **F1 Score** | **80.95%** |
| Confusion Matrix | TP=51, TN=125, FP=14, FN=10 |

---

### 2. K-Nearest Neighbors (KNN)
> Instance-based learning with K=5 neighbors.

| Metric | Score |
|---|---|
| Accuracy | 78.5% |
| Precision | 67.31% |
| Recall | 57.38% |
| F1 Score | 61.95% |
| Confusion Matrix | TP=35, TN=122, FP=17, FN=26 |

---

### 3. Naive Bayes (Gaussian NB)
> Probabilistic classifier based on Bayes' theorem with Gaussian distribution assumption.

| Metric | Score |
|---|---|
| Accuracy | 86.0% |
| Precision | 81.13% |
| Recall | 70.49% |
| F1 Score | 75.44% |
| Confusion Matrix | TP=43, TN=129, FP=10, FN=18 |

---

## 📊 Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **88.0%** ⭐ | 78.46% | 83.61% | **80.95%** ⭐ |
| Naive Bayes | 86.0% | **81.13%** | 70.49% | 75.44% |
| KNN (k=5) | 78.5% | 67.31% | 57.38% | 61.95% |

> 🏆 **Best Model**: Logistic Regression — highest accuracy (88%) and best F1 score (80.95%)

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML Models | `scikit-learn` |
| Feature Scaling | `StandardScaler` |
| Evaluation | `confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score` |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. Open the notebook
jupyter notebook "12 credit_wise.ipynb"
```

---

## 📈 Workflow

```
Raw Data → EDA → Missing Value Imputation → Feature Engineering
→ Encoding → Train/Test Split → StandardScaler
→ Train Models (LR / KNN / Naive Bayes) → Evaluate → Compare → Best Model
```

---

## 👨‍💻 Author

**Muhammad Amman** | AI/ML Project | Anaconda + Jupyter Notebook
