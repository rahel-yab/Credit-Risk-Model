from pydantic import BaseModel
from typing import Optional

class TransactionData(BaseModel):
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    PricingStrategy: int

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool