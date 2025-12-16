# Credit Risk Modeling Project

## Project Overview

This project focuses on building a **credit risk modeling pipeline** using transactional data in a context where **no explicit default labels are available**. The primary objective is to estimate customer credit risk by engineering a **proxy target variable** derived from customer behavior, enabling supervised learning in alignment with real-world credit scoring constraints.

The project follows industry-aligned best practices, including exploratory data analysis (EDA), feature engineering, proxy target construction, model training, testing, and production deployment.

---

## Business Understanding 

### Credit Risk and Credit Scoring Context

Credit risk refers to the likelihood that a customer will fail to meet their financial obligations. Financial institutions rely on **credit scoring models** to quantify this risk and support decisions such as loan approval, pricing, and credit limits.

In traditional credit risk problems, a binary **default label** is available. However, in this challenge, no such label exists. This mirrors real-world scenarios where historical default data may be incomplete or unavailable, requiring alternative strategies to approximate risk.

---

### Basel II Regulatory Context

Under the **Basel II Capital Accord**, financial institutions are required to:

- Quantify credit risk using internal or standardized models  
- Hold sufficient capital proportional to risk exposure  
- Ensure models are interpretable, auditable, and data-driven  

A key implication of Basel II is that **risk estimation must be justifiable and explainable**, even when direct default labels are absent. This project addresses that requirement by constructing a behavior-based proxy target grounded in established credit risk theory.

---

### Why a Proxy Target Variable Is Required

The dataset does **not contain a direct default or credit outcome variable**. Using unrelated labels (e.g., fraud indicators) would violate the business objective and produce misleading models.

To address this, a **proxy target variable** is engineered based on the assumption that **customer transactional behavior reflects underlying creditworthiness**. This approach is widely used in early-stage credit modeling and exploratory risk assessment.

---

### Proxy Target Construction Strategy

The proxy target is constructed using **RFM analysis and clustering**:

- **Recency**: Time since the customer’s most recent transaction  
- **Frequency**: Number of transactions over a defined period  
- **Monetary**: Total or average transaction value  

Customers are aggregated at the customer level and clustered using **K-Means**. Clusters exhibiting low engagement, long inactivity, or extreme/irregular behavior are labeled as **higher risk**, while consistent and active customers are labeled as **lower risk**.

This cluster-derived label becomes the supervised learning target.

---

## Dataset Description

The dataset consists of transactional records with the following feature groups:

- Identifiers: TransactionId, CustomerId, AccountId, SubscriptionId  
- Transaction values: Amount, Value  
- Product information: ProductId, ProductCategory, ProviderId  
- Channel information: ChannelId  
- Temporal features: TransactionStartTime  

The raw data is transformed into customer-level features for modeling.

---

## Exploratory Data Analysis (EDA)

EDA is conducted in `notebooks/eda.ipynb` and focuses on understanding data quality, behavior patterns, and modeling feasibility.

### EDA Scope

- Dataset structure and data types  
- Summary statistics for numerical variables  
- Distribution of categorical features  
- Missing-value analysis  
- Outlier detection  
- Behavioral insights relevant to credit risk  

---

### Missing Value Analysis

Missing values were explicitly analyzed across all features:

- Core transactional and customer identifier fields are complete  
- No critical missing values prevent customer-level aggregation  
- Identifier columns are retained only for grouping and dropped before modeling  
- No imputation is required for monetary or temporal features  

This confirms that the dataset is suitable for behavioral aggregation without introducing bias from missing data handling.

---

### Key EDA Insights

- Transaction amounts are highly skewed, supporting aggregation over raw usage  
- Individual transactions are weak risk indicators; **customer-level behavior is essential**  
- Product category and channel usage patterns differ significantly across customers  
- RFM features provide meaningful separation of behavioral risk groups  

EDA insights directly inform feature engineering and proxy target construction.

---

## Feature Engineering

Feature engineering focuses on transforming transaction-level data into meaningful customer-level predictors.

### Implemented Feature Types

- **Aggregated features**: transaction count, total amount, average value, standard deviation
- **Temporal features**: recency, transaction velocity, activity trends
- **Behavioral diversity**: number of unique channels and product categories
- **Categorical encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

---

### WoE and IV Implementation

Weight of Evidence (WoE) encoding is applied to categorical features to improve interpretability and model stability. Information Value (IV) guides feature selection and risk signal strength assessment.

---

## Proxy Target Variable Engineering

### RFM Analysis Implementation

- **Recency**: Days since last transaction (calculated from snapshot date)
- **Frequency**: Total number of transactions per customer
- **Monetary**: Total transaction value per customer

### Clustering Approach

- K-Means clustering (k=3) applied to standardized RFM features
- Cluster 0 identified as high-risk (lowest engagement: low frequency, low monetary)
- Binary target `is_high_risk` assigned: 1 for high-risk cluster, 0 for others

---

## Modeling Approach

### Implemented Models

- **Logistic Regression** (primary model - interpretable and Basel II compliant)
- Class imbalance handling through weighting and stratification
- Hyperparameter tuning with GridSearchCV

### Model Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Performance**: ROC-AUC > 0.99, F1-Score > 0.92 on test set
- **Cross-validation**: Stratified k-fold validation  

---

## Experiment Tracking and Reproducibility

**MLflow** is fully implemented for:

- ✅ Experiment tracking
- ✅ Hyperparameter logging
- ✅ Metric comparison
- ✅ Model artifact management
- ✅ Model registry for production deployment

This ensures transparency and reproducibility consistent with production credit risk systems.

---

## Production Deployment

### FastAPI REST API

- **Endpoint**: `/predict` - accepts transaction data, returns risk probability
- **Data validation**: Pydantic models for request/response schemas
- **Model loading**: Loads registered model from MLflow Model Registry
- **Documentation**: Automatic OpenAPI/Swagger docs at `/docs`

### Containerization

- **Docker**: Containerized API with all dependencies
- **Docker Compose**: Orchestrated deployment with MLflow server
- **Environment**: Consistent deployment across development and production

### CI/CD Pipeline

- **GitHub Actions**: Automated testing and linting
- **Triggers**: Push to main/task branches and pull requests
- **Checks**: flake8 linting, pytest execution
- **Quality Gates**: Pipeline fails on linting or test failures

---

## Project Structure

```text
credit-risk-model/
│
├── data/
│   ├── raw/                  # raw data
│   └── processed/            # processes data
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering & proxy target
│   ├── predict.py             # Inference utilities
│   ├── train.py               # Model training with MLflow
│   ├── __pycache__/
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # API data models
│
├── tests/
│   ├── test_data_processing.py
│   └── __pycache__/
│
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
│
├── Dockerfile                 # API containerization
├── docker-compose.yml         # Orchestrated deployment
├── requirements.txt           # Python dependencies
├── README.md
└── venv/                      # Virtual environment
```
---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahel-yab/Credit-Risk-Model.git
   cd credit-risk-model
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run training pipeline**
   ```bash
   python src/train.py
   ```

4. **Deploy API**
   ```bash
   docker compose up --build
   ```

### API Usage

Once deployed, the API is available at `http://localhost:8000`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m '\''Add amazing feature'\'')
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request