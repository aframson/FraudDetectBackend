# type:ignore

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import joblib
import motor.motor_asyncio
from geopy.distance import geodesic
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and scaler
model = joblib.load('Random_SMOTE_Forest.joblib')
scaler = joblib.load('scaler.joblib')

# MongoDB setup
client = motor.motor_asyncio.AsyncIOMotorClient('mongodb+srv://obiriaframson:Ra2aDt6BrogRfLKT@cluster0.dusaw98.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['transaction_db']
transactions_collection = db['transactions']
users_collection = db['users']

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Secret
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class User(BaseModel):
    username: str
    email: str
    password: str
    dob: str

class LoginUser(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str

class Transaction(BaseModel):
    cc_num: int
    amt: float
    trans_date: str
    latitude: float
    longitude: float
    dob: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_user(username: str):
    user = await users_collection.find_one({"username": username})
    return user

async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    if user and verify_password(password, user['password']):
        return user
    return False

async def save_user(user: dict):
    await users_collection.insert_one(user)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/register", response_model=Token)
async def register(user: User):
    hashed_password = get_password_hash(user.password)
    user_dict = {"username": user.username, "email": user.email, "password": hashed_password, "dob": user.dob}
    await save_user(user_dict)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
async def login(user: LoginUser):
    user = await authenticate_user(user.username, user.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

def calculate_age(dob, trans_date):
    dob_date = datetime.strptime(dob, '%Y-%m-%d')
    age = trans_date.year - dob_date.year - ((trans_date.month, trans_date.day) < (dob_date.month, dob_date.day))
    return age

async def calculate_time_diff(cc_num: int, trans_date: datetime):
    last_transaction = await transactions_collection.find_one(
        {'cc_num': cc_num},
        sort=[('trans_date', -1)]
    )
    if last_transaction:
        last_trans_date = last_transaction['trans_date']
        time_diff = (trans_date - last_trans_date).total_seconds()
    else:
        time_diff = 0
    return time_diff

def calculate_distance_to_merchant(lat, long):
    transaction_distance = (lat, long)
    merchant_distance = (23.67485, -60.73840)
    return geodesic(transaction_distance, merchant_distance).km

def is_anomalous(distance_diff, time_diff, distance_threshold=100, time_threshold=3600):
    return distance_diff > distance_threshold or time_diff < time_threshold

async def save_transaction(transaction: dict):
    await transactions_collection.insert_one(transaction)

@app.post("/predict") 
async def predict(transaction: Transaction):
    try:
        trans_date = datetime.strptime(str(transaction.trans_date).replace('T',' '), '%Y-%m-%d %H:%M')
    except ValueError:
        trans_date = datetime.strptime(str(transaction.trans_date).replace('T',' '), '%Y-%m-%d %H:%M')

    age = calculate_age(transaction.dob, trans_date)
    time_diff = await calculate_time_diff(transaction.cc_num, trans_date)
    distance_bn_trans = calculate_distance_to_merchant(transaction.latitude, transaction.longitude)

    last_4 = str(transaction.cc_num)[-4:]
    bin = str(transaction.cc_num)[:6]
    

    is_anomaly = int(is_anomalous(distance_bn_trans, time_diff))

    new_data = {
        'gender': 1,
        'day': trans_date.day,
        'year': trans_date.year,
        'hour': trans_date.hour,
        'minute': trans_date.minute,
        'age': age,
        'cc_num_last_4': last_4,
        'bin': bin,
        'distance_bn_trans': distance_bn_trans,
        'time_diff': time_diff,
        'is_anomaly': is_anomaly
    }

    new_df = pd.DataFrame([new_data])
    feature_columns = ['gender', 'day', 'year', 'hour', 'minute', 'age', 'cc_num_last_4', 'bin', 'distance_bn_trans', 'time_diff', 'is_anomaly']
    new_df = new_df[feature_columns]
    new_df_scaled = scaler.transform(new_df)

    prediction = model.predict(new_df_scaled)[0]
    predicted_probabilities = model.predict_proba(new_df_scaled)[0]

    fraud_class = {
        0: 'Not Fraud',
        1: 'Fraud'
    }

    transaction_dict = transaction.dict()
    transaction_dict['trans_date'] = trans_date
    transaction_dict['time_diff'] = time_diff
    transaction_dict['status'] = fraud_class[prediction]
    transaction_dict['probability'] = predicted_probabilities.tolist()
    transaction_dict['distance_bn_trans'] = distance_bn_trans
    transaction_dict['is_anomaly'] = is_anomaly

    await save_transaction(transaction_dict)

    return {
        "Result": fraud_class[prediction],
        "predicted_class": int(prediction),
        "predicted_probabilities": predicted_probabilities.tolist()
    }

@app.get("/transactions", response_model=List[Dict])
async def get_transactions(current_user: User = Depends(get_current_user)):
    transactions = []
    async for transaction in transactions_collection.find():
        transaction['_id'] = str(transaction['_id'])
        transactions.append(transaction)
    return transactions
