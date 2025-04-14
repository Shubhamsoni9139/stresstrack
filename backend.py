from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

app = FastAPI()

LIVE_DATA_PATH = "live_training_data.csv"
ACCURACY_LOG_PATH = "accuracy.csv"
MODEL_DIR = "saved_models"

# Ensure files exist
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(LIVE_DATA_PATH):
    pd.DataFrame(
        columns=[
            "HR",
            "respr",
            "Time_sec",
            "random_forest",
            "logistic_regression",
            "ensemble",
        ]
    ).to_csv(LIVE_DATA_PATH, index=False)

if not os.path.exists(ACCURACY_LOG_PATH):
    pd.DataFrame(
        columns=["timestamp", "random_forest", "logistic_regression", "ensemble"]
    ).to_csv(ACCURACY_LOG_PATH, index=False)

# Load models
rf_model = joblib.load(f"{MODEL_DIR}/random_forest_model.pkl")
lr_model = joblib.load(f"{MODEL_DIR}/logistic_regression_model.pkl")
ensemble_model = joblib.load(f"{MODEL_DIR}/ensemble_model.pkl")


# Define input schema
class InputData(BaseModel):
    HR: float
    respr: float
    Time_sec: int


@app.post("/predict")
def predict(data: InputData):
    try:
        input_features = [[data.HR, data.respr]]
        rf_pred = int(rf_model.predict(input_features)[0])
        lr_pred = int(lr_model.predict(input_features)[0])
        ensemble_pred = int(ensemble_model.predict(input_features)[0])

        new_row = pd.DataFrame(
            [
                {
                    "HR": data.HR,
                    "respr": data.respr,
                    "Time_sec": data.Time_sec,
                    "random_forest": rf_pred,
                    "logistic_regression": lr_pred,
                    "ensemble": ensemble_pred,
                }
            ]
        )
        new_row.to_csv(LIVE_DATA_PATH, mode="a", header=False, index=False)

        # Accuracy calculation
        df = pd.read_csv(LIVE_DATA_PATH)
        label_column = "Label" if "Label" in df.columns else "ensemble"
        accuracies = {}
        for model_col in ["random_forest", "logistic_regression", "ensemble"]:
            if label_column in df.columns and model_col in df.columns:
                acc = accuracy_score(df[label_column], df[model_col])
                accuracies[model_col] = round(acc, 4)

        return {
            "predictions": {
                "random_forest": rf_pred,
                "logistic_regression": lr_pred,
                "ensemble": ensemble_pred,
            },
            "accuracies": accuracies,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background retraining function
def retrain_models_periodically():
    while True:
        try:
            time.sleep(30)
            df = pd.read_csv(LIVE_DATA_PATH)

            if "Label" not in df.columns or len(df) < 20:
                continue

            X = df[["HR", "respr"]]
            y = df["Label"]

            # Train models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")

            rf.fit(X, y)
            lr.fit(X, y)

            ensemble = VotingClassifier(
                estimators=[("rf", rf), ("lr", lr)], voting="hard"
            )
            ensemble.fit(X, y)

            # Save models
            joblib.dump(rf, f"{MODEL_DIR}/random_forest_model.pkl")
            joblib.dump(lr, f"{MODEL_DIR}/logistic_regression_model.pkl")
            joblib.dump(ensemble, f"{MODEL_DIR}/ensemble_model.pkl")

            # Update global models
            global rf_model, lr_model, ensemble_model
            rf_model, lr_model, ensemble_model = rf, lr, ensemble

            # Log accuracy
            rf_acc = accuracy_score(y, rf.predict(X))
            lr_acc = accuracy_score(y, lr.predict(X))
            ensemble_acc = accuracy_score(y, ensemble.predict(X))

            accuracy_row = pd.DataFrame(
                [
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "random_forest": round(rf_acc, 4),
                        "logistic_regression": round(lr_acc, 4),
                        "ensemble": round(ensemble_acc, 4),
                    }
                ]
            )

            accuracy_row.to_csv(
                ACCURACY_LOG_PATH,
                mode="a",
                header=not os.path.exists(ACCURACY_LOG_PATH),
                index=False,
            )

            print(
                f"[{time.strftime('%X')}] ✅ Models retrained and saved. Accuracy logged."
            )

        except Exception as e:
            print(f"[Error] during retraining: {e}")


# Start background thread
threading.Thread(target=retrain_models_periodically, daemon=True).start()
