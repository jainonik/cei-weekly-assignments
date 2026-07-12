# Lead Scoring – Logistic Regression

> **Predicting which leads are most likely to convert for X Education**

---

## Project Overview

X Education sells online courses and generates many leads daily. However, only ~30% of leads convert to paying customers. This project builds a **Lead Scoring Model** using Logistic Regression to assign a score (0–100) to each lead — enabling the sales team to prioritize high-potential leads and achieve a target conversion rate of ~80%.

---

## Project Structure

```
Documented/
├── lead-scoring-documented.ipynb   # Main notebook (fully documented)
├── Lead Scoring.csv                # Raw dataset
├── lead scores.csv                 # Model output – lead scores
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## Dataset

| Property        | Details                              |
|-----------------|--------------------------------------|
| **File**        | `Lead Scoring.csv`                   |
| **Rows**        | ~9,000 leads                         |
| **Target**      | `Converted` (1 = converted, 0 = not) |
| **Features**    | Demographics, activity, source, etc. |
| **Provided by** | X Education                          |

---

## Methodology

### Step-by-Step Pipeline

| Step | Description |
|------|-------------|
| 1. Data Loading | Load raw CSV; inspect shape and dtypes |
| 2. Data Cleaning | Drop high-null columns (>3000 missing); impute remaining with mode/median |
| 3. Standardization | Fix inconsistent category values (e.g. `google` → `Google`) |
| 4. EDA | Univariate & bivariate analysis; pairplot, heatmap, countplots |
| 5. Feature Engineering | Square-root transformation to reduce skewness |
| 6. Outlier Removal | 99.7th percentile capping (2 iterations) |
| 7. Encoding | One-Hot Encoding of all categorical variables |
| 8. Train-Test Split | 70% train / 30% test (random_state=42) |
| 9. Standardization | StandardScaler (Z-score normalization) |
| 10. Baseline Model | Logistic Regression on all features |
| 11. Feature Selection | VIF analysis – keep features with VIF ≤ 5 |
| 12. Final Model | GLM (Binomial) on VIF-adjusted features |
| 13. Threshold Tuning | Optimal cutoff = 0.375 (sensitivity-specificity balance) |
| 14. Lead Scoring | Score = predicted probability × 100 |

---

## Model Performance

| Metric | Baseline Model | VIF-Adjusted Model |
|--------|---------------|-------------------|
| Accuracy | ~80% | ~80% |
| Precision | — | High |
| Recall (Sensitivity) | — | Balanced |
| ROC-AUC | > 0.85 | > 0.85 |
| No. of Features | All | Reduced (VIF ≤ 5) |

---

## Lead Score Interpretation

| Score Range | Priority | Recommended Action |
|-------------|----------|--------------------|
| 75 – 100 | 🔴 High | Immediate follow-up call |
| 50 – 74 | 🟡 Medium | Nurture via email campaign |
| 25 – 49 | 🟢 Low | Automated drip emails |
| 0 – 24 | ⚪ Cold | No immediate action needed |

---

## Key Business Insights

- **Leads with high website time** are the strongest converters
- **Working professionals** convert at a significantly higher rate
- **Google and direct traffic** are the highest-quality lead sources
- **SMS Sent / Email Opened** as last activity strongly predicts conversion
- Focusing on top-scored leads can improve conversion rate from **30% → 80%**

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
# Option 1: Jupyter Notebook
jupyter notebook lead-scoring-documented.ipynb

# Option 2: Execute all cells automatically
python -m jupyter nbconvert --to notebook --execute lead-scoring-documented.ipynb --output lead-scoring-documented.ipynb
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical visualization |
| `scipy` | Statistical tests, skewness |
| `scikit-learn` | ML models, preprocessing, metrics |
| `statsmodels` | GLM, VIF analysis |

---

## Algorithm Details

- **Model:** Logistic Regression (Binary Classification)
- **Solver:** `liblinear` (efficient for small/medium datasets)
- **Family:** Binomial (via `statsmodels.GLM`)
- **Feature Selection:** Variance Inflation Factor (VIF ≤ 5)
- **Optimal Threshold:** 0.375 (determined via sensitivity-specificity analysis)

---

*Developed as part of X Education Lead Scoring Case Study*
