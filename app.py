from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi import Request

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://customer-churn-prediction-frontend-two.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter



class ChurnInput(BaseModel):
    Tenure: float
    CityTier: int
    WarehouseToHome: float
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    SatisfactionScore: int
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float

    PreferredLoginDevice_Mobile_Phone: bool
    PreferredLoginDevice_Phone: bool
    PreferredPaymentMode_COD: bool
    PreferredPaymentMode_Cash_on_Delivery: bool
    PreferredPaymentMode_Credit_Card: bool
    PreferredPaymentMode_Debit_Card: bool
    PreferredPaymentMode_E_wallet: bool
    PreferredPaymentMode_UPI: bool
    Gender_Male: bool
    PreferedOrderCat_Grocery: bool
    PreferedOrderCat_Laptop_and_Accessory: bool
    PreferedOrderCat_Mobile: bool
    PreferedOrderCat_Mobile_Phone: bool
    PreferedOrderCat_Others: bool
    MaritalStatus_Married: bool
    MaritalStatus_Single: bool

model = joblib.load("churn_model_pipeline.pkl")

@app.post("/predict")
@limiter.limit("10/minute")
def predict_churn(request: Request, data: ChurnInput): 
    input_dict = {
        "Tenure": data.Tenure,
        "CityTier": data.CityTier,
        "WarehouseToHome": data.WarehouseToHome,
        "HourSpendOnApp": data.HourSpendOnApp,
        "NumberOfDeviceRegistered": data.NumberOfDeviceRegistered,
        "SatisfactionScore": data.SatisfactionScore,
        "NumberOfAddress": data.NumberOfAddress,
        "Complain": data.Complain,
        "OrderAmountHikeFromlastYear": data.OrderAmountHikeFromlastYear,
        "CouponUsed": data.CouponUsed,
        "OrderCount": data.OrderCount,
        "DaySinceLastOrder": data.DaySinceLastOrder,
        "CashbackAmount": data.CashbackAmount,
        "PreferredLoginDevice_Mobile Phone": data.PreferredLoginDevice_Mobile_Phone,
        "PreferredLoginDevice_Phone": data.PreferredLoginDevice_Phone,
        "PreferredPaymentMode_COD": data.PreferredPaymentMode_COD,
        "PreferredPaymentMode_Cash on Delivery": data.PreferredPaymentMode_Cash_on_Delivery,
        "PreferredPaymentMode_Credit Card": data.PreferredPaymentMode_Credit_Card,
        "PreferredPaymentMode_Debit Card": data.PreferredPaymentMode_Debit_Card,
        "PreferredPaymentMode_E wallet": data.PreferredPaymentMode_E_wallet,
        "PreferredPaymentMode_UPI": data.PreferredPaymentMode_UPI,
        "Gender_Male": data.Gender_Male,
        "PreferedOrderCat_Grocery": data.PreferedOrderCat_Grocery,
        "PreferedOrderCat_Laptop & Accessory": data.PreferedOrderCat_Laptop_and_Accessory,
        "PreferedOrderCat_Mobile": data.PreferedOrderCat_Mobile,
        "PreferedOrderCat_Mobile Phone": data.PreferedOrderCat_Mobile_Phone,
        "PreferedOrderCat_Others": data.PreferedOrderCat_Others,
        "MaritalStatus_Married": data.MaritalStatus_Married,
        "MaritalStatus_Single": data.MaritalStatus_Single
    }
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(probability, 4)
    }

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
