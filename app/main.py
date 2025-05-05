from fastapi import FastAPI, HTTPException, status
import numpy as np
import os
from dotenv import load_dotenv
from .schemas import FeatureVector
from sqlalchemy import create_engine, text

load_dotenv()
app = FastAPI()


def get_model_parameters() -> tuple[np.ndarray | float, ...]:
    """Retrieves model parameters from database

    Returns:
        tuple[np.ndarray | float, ...]: weights, bias, training mean, standard deviation of latest trained model
    """
    db_url = f"postgresql+psycopg2://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"

    engine = create_engine(db_url)
    query = text("""
        SELECT weights, bias, training_mean, training_std 
        FROM model_params 
        ORDER BY trained_at 
        DESC LIMIT 1;             
    """)

    with engine.connect() as conn:
        w, b, train_mean, train_std = conn.execute(query).fetchone()

    return np.array(w), b, np.array(train_mean), np.array(train_std)


@app.post("/predict", status_code=status.HTTP_201_CREATED)
def predict(features: FeatureVector):
    """Endpoint that allows users to pass in data to get an estimated house price back

    Args:
        features (FeatureVector): json payload of information that makes up the feature vector

    Returns:
        dict: json payload containing the predicted house price
    """
    try:
        w, b, train_mean, train_std = get_model_parameters()
        X_pred: np.ndarray = np.array(
            [
                features.crim,
                features.zn,
                features.indus,
                features.nox,
                features.rm,
                features.age,
                features.dis,
                features.rad,
                features.tax,
                features.ptratio,
                features.b,
                features.lstat,
                features.chas,
            ]
        )
        X_pred = (X_pred - train_mean) / train_std
        y_hat: float = np.dot(X_pred, w) + b
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction": y_hat}


@app.get("/")
def health():
    return {"message": "Successful"}
