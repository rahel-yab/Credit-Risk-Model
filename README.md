# Credit Risk Modeling Using Transactional Data

## Project Overview

This project implements an end-to-end credit risk and fraud modeling pipeline using transactional data. The objective is to transform raw transaction records into a reliable, reproducible machine learning system that can estimate risk probabilities and support credit decision-making.

The work emphasizes strong data understanding, transparent preprocessing, interpretable baseline modeling, and production-oriented structure. The project is designed to evolve toward a deployable credit risk scoring service.

---

## Business Context

In a financial services environment, especially credit and fraud risk, model decisions must be explainable, auditable, and robust to data quality issues. This project focuses on:

- Translating transactional behavior into predictive signals
- Handling mixed data types (numerical, categorical, identifiers)
- Building reproducible pipelines suitable for regulated contexts
- Laying groundwork for deployment and monitoring

---


## Project Structure

```text
credit-risk-model/
│
├── data/
│ ├── raw/ # Original, immutable dataset
│ └── processed/ # Cleaned and model-ready datasets
│
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
│
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering & preprocessing
│ └── train.py # Model training pipeline
│
├── tests/
│ └── test_data_processing.py # Unit tests
│
├── models/
│ └── logistic_regression.pkl # Saved trained model
│
├── requirements.txt
├── README.md
└── venv/

```
---


---

## Dataset Description

The dataset consists of transaction-level records with the following categories:

- **Identifiers**  
  TransactionId, BatchId, AccountId, SubscriptionId, CustomerId

- **Monetary Features**  
  Amount, Value

- **Categorical Features**  
  CurrencyCode, CountryCode, ProviderId, ProductId, ProductCategory, ChannelId

- **Temporal Feature**  
  TransactionStartTime

- **Target Variable**  
  FraudResult (binary indicator)

Raw data is stored under `data/raw/`. All datasets used for modeling are derived programmatically and saved under `data/processed/`.

---

## Data Quality and Missing-Value Handling

Data quality checks were performed during EDA and preprocessing.

### Missing Values
- Core monetary features showed minimal missingness.
- Rows with missing target values were excluded from modeling.
- Categorical variables are handled using encoders that safely ignore unseen or missing categories at inference time.

### Outliers
- Transaction amounts and values exhibit heavy skew.
- Extreme values were retained, as they may carry meaningful fraud or risk signals rather than noise.

All handling decisions are implemented in code to ensure full reproducibility.

---

## Exploratory Data Analysis (EDA)

EDA is conducted in `notebooks/eda.ipynb` and focuses on:

- Dataset structure and data types
- Distributions of numerical and categorical variables
- Identification of missing values and potential outliers
- Correlations between monetary behavior and fraud outcomes

### Key Insights from EDA
- Fraud cases are rare, indicating strong class imbalance.
- Certain product categories and channels show higher fraud prevalence.
- Monetary features are highly skewed, motivating feature scaling.
- Identifier columns provide no predictive signal and must be excluded.

These insights directly informed feature selection, preprocessing, and model design.

---

## Feature Engineering and Preprocessing

Feature engineering is implemented using scikit-learn Pipelines and ColumnTransformers to ensure consistency between training and inference.

Key steps include:
- Dropping identifier columns
- Separating numerical and categorical features
- Standardizing numerical features
- One-hot encoding categorical variables
- Integrating preprocessing directly into the model pipeline

This approach prevents data leakage and ensures production safety.

---

## Modeling Approach

A baseline Logistic Regression model is used as the first benchmark due to its interpretability and suitability for regulated environments.

Model characteristics:
- Pipeline-based preprocessing + modeling
- Stratified train-test split
- Class-weighted learning to address imbalance
- Evaluation using accuracy, precision, recall, and ROC-AUC
- Serialized model artifact saved for reuse

This baseline establishes a clear performance reference for future model improvements.

---

## Testing

Unit tests are implemented using `pytest` to validate core data processing logic.

Tests ensure:
- Feature engineering functions return expected outputs
- Preprocessing pipelines are constructed correctly

Testing improves maintainability and guards against silent failures.

---

## Deployment and Operationalization (Current Status)

Currently implemented:
- Script-based model training (`python -m src.train`)
- Serialized model artifacts stored under `models/`
- Reproducible preprocessing embedded in the pipeline

Planned extensions:
- Inference script (`predict.py`)
- Containerization (Docker)
- CI/CD integration
- Monitoring and retraining hooks

These components will be added in later stages.

---

## How to Get Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Dataset
Place the raw dataset file into:
```bash
data/raw/
```
### 5. Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/eda.ipynb
```

### 6. Train the Model
```bash
python -m src.train
```

### 7. Run Tests
```bash
pytest tests/
```


# Contribution Guidelines

Contributions are welcome and encouraged.

To contribute:

Fork the repository

Create a feature branch

Follow existing coding and documentation standards

Add or update unit tests where appropriate

Submit a pull request with a clear description of changes

All contributions should prioritize reproducibility, clarity, and correctness.
