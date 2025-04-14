import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load the data
df = pd.read_csv("stress_dataset.csv")

# Clean and rename column if needed
df.dropna(inplace=True)
df.rename(columns={"Time(sec)": "Time_sec"}, inplace=True)

# Features and labels
X = df[["HR", "respr"]]  # You can also include 'Time_sec' if it's meaningful
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(max_iter=1000)

# Train base models
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Ensemble model using soft voting
ensemble_model = VotingClassifier(
    estimators=[("rf", rf_model), ("lr", lr_model)], voting="soft"
)

ensemble_model.fit(X_train, y_train)

# Evaluation
models = {
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "Ensemble": ensemble_model,
}

for name, model in models.items():
    preds = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.2f}")
    print(classification_report(y_test, preds))

# === Save models to disk ===
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(model_dir, "random_forest_model.pkl"))
joblib.dump(lr_model, os.path.join(model_dir, "logistic_regression_model.pkl"))
joblib.dump(ensemble_model, os.path.join(model_dir, "ensemble_model.pkl"))

print("\n✅ Models saved in the 'saved_models' folder.")
