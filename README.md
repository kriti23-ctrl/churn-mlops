# Churn Prediction MLOps Pipeline

An end-to-end ML pipeline to predict customer churn using Random Forest and XGBoost, with experiment tracking and a REST API.

## Live API
https://churn-mlops-82cq.onrender.com/docs

## Tech Stack
- Python, pandas, scikit-learn, XGBoost
- MLflow (experiment tracking)
- FastAPI + Uvicorn (model serving)
- Render (deployment)

## Results
| Model | F1 Score | Accuracy |
|-------|----------|----------|
| Random Forest | 0.5562 | 0.7925 |
| XGBoost | 0.5235 | 0.7697 |

## How to Run
```bash
python src/train.py       # Train models
uvicorn api.main:app      # Start API
mlflow ui                 # View experiments
```

## API
POST /predict → returns churn prediction and probability
