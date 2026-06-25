import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "RandomForest")
    print(f"RandomForest -> F1: {f1:.4f}, Accuracy: {acc:.4f}")

with mlflow.start_run(run_name="XGBoost"):
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    mlflow.log_param("model", "XGBoost")
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)
    mlflow.xgboost.log_model(xgb, "XGBoost")
    print(f"XGBoost -> F1: {f1:.4f}, Accuracy: {acc:.4f}")

joblib.dump(xgb, "model.pkl")
print("Model saved to model.pkl")
