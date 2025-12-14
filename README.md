# Credit Risk Probability Model Using Alternative Data

## Project Overview
Bati Bank is partnering with an eCommerce platform to offer a Buy-Now-Pay-Later (BNPL) service.  
The core challenge is to assess credit risk for customers who lack traditional credit histories.

This project builds an end-to-end credit risk modeling pipeline using **alternative behavioral data** derived from transaction logs. By transforming transaction patterns into structured risk signals, the system produces:
- A probability of default proxy
- A credit score derived from that probability
- Model outputs that can support loan approval decisions, limits, and durations

The solution is designed with **reproducibility, interpretability, and deployment-readiness** in mind.

---

## Credit Scoring Business Understanding

### Basel II and Model Interpretability
The Basel II Capital Accord emphasizes risk-sensitive capital allocation and requires financial institutions to demonstrate transparency, documentation, and explainability in their risk models.  

As a result, our credit scoring model must:
- Be interpretable and auditable
- Use well-defined assumptions and proxy definitions
- Allow regulators and risk officers to understand how inputs affect outputs

This influences both model choice and feature engineering decisions, favoring approaches that balance predictive power with clarity.

---

### Why a Proxy Target Variable Is Necessary
The dataset does not contain a direct label indicating whether a customer defaulted on credit. Without this, supervised learning is not directly possible.

To address this, we define a **proxy target variable** based on customer engagement behavior using Recency, Frequency, and Monetary (RFM) analysis. Customers who exhibit low engagement are treated as higher-risk proxies.

**Business risks of using a proxy include:**
- Misclassification of customers who are inactive but creditworthy
- Bias introduced by behavioral patterns unrelated to repayment ability
- Proxy drift as customer behavior changes over time

These risks are mitigated by:
- Clear documentation of assumptions
- Conservative model usage
- Continuous monitoring and retraining

---

### Model Complexity vs Interpretability Trade-offs
In regulated financial environments, there is a trade-off between model transparency and predictive performance.

| Approach | Advantages | Limitations |
|--------|-----------|-------------|
| Logistic Regression with WoE | Highly interpretable, regulator-friendly, stable | Lower ceiling on performance |
| Tree-based Models (GBM, RF) | Capture non-linear relationships, higher accuracy | Harder to explain, risk of overfitting |

This project evaluates multiple models and selects a final model based on **performance, stability, and governance suitability**, not accuracy alone.

---

## Data Description
The dataset comes from the Xente eCommerce platform and includes transaction-level records with:
- Customer identifiers
- Transaction timestamps
- Monetary values
- Product and channel metadata
- Fraud indicators

Raw data is stored in `data/raw/` and processed datasets are stored in `data/processed/`.

---

## Project Structure

```text
credit-risk-model/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
---

## Exploratory Data Analysis (EDA)

EDA is performed in notebooks/eda.ipynb and focuses on:

Data structure and types

Distribution of numerical and categorical variables

Missing values and outliers

Correlations between monetary and behavioral features

---

## Feature Engineering

Feature engineering is implemented in src/data_processing.py using reproducible pipelines.

Key transformations include:
    Aggregated customer-level metrics (total amount, average amount, transaction count, variability)

- Temporal features extracted from transaction timestamps
- Categorical encoding using One-Hot Encoding
- Numerical scaling
- Weight of Evidence (WoE) transformations for selected features

All transformations are chained using sklearn.pipeline.Pipeline.

---

## Proxy Target Variable Construction

Since no default label exists, a proxy is created using RFM analysis:

Compute Recency, Frequency, and Monetary metrics per customer

Scale RFM features

Apply K-Means clustering (k=3)

Identify the least engaged cluster

Assign is_high_risk = 1 to that cluster, 0 otherwise

This proxy target is merged back into the modeling dataset.

 ---

## Model Training and Experiment Tracking

Model training is handled in src/train.py.

Key steps:
- Train/test split with fixed random state
- Multiple models trained and evaluated
- Hyperparameter tuning using Grid Search or Random Search

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Testing

Unit tests are implemented using pytest and located in tests/.

Tests validate:
- Feature engineering outputs
- Pipeline consistency
- Presence of expected columns
- Tests are executed automatically as part of the CI pipeline.
