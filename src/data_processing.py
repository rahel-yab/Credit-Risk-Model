import pandas as pd

df = pd.read_csv("../data/processed/transactions_with_target.csv")
df.head()

df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
df["transaction_hour"] = df["TransactionStartTime"].dt.hour
df["transaction_day"] = df["TransactionStartTime"].dt.day
df["transaction_month"] = df["TransactionStartTime"].dt.month
df["transaction_year"] = df["TransactionStartTime"].dt.year

df[[
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year"
]].head()


customer_features = (
    df.groupby("CustomerId")
    .agg(
        total_transaction_value=("Value", "sum"),
        avg_transaction_value=("Value", "mean"),
        transaction_count=("TransactionId", "count"),
        std_transaction_value=("Value", "std"),
        most_common_channel=("ChannelId", lambda x: x.mode()[0]),
        most_common_category=("ProductCategory", lambda x: x.mode()[0]),
        is_high_risk=("is_high_risk", "max")
    )
    .reset_index()
)

customer_features.head()


# Handle Missing Values
customer_features.isnull().sum()
customer_features["std_transaction_value"] = customer_features[
    "std_transaction_value"
].fillna(0)
# Separate Features and Target
X = customer_features.drop(columns=["CustomerId", "is_high_risk"])
y = customer_features["is_high_risk"]


# identify feature types

categorical_features = [
    "most_common_channel",
    "most_common_category"
]

numerical_features = [
    "total_transaction_value",
    "avg_transaction_value",
    "transaction_count",
    "std_transaction_value"
]

categorical_features, numerical_features
