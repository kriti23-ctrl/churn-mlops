import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    
    # Drop customerID column
    df.drop("customerID", axis=1, inplace=True)
    
    # Fix TotalCharges column (has spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    
    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Encode all categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    print("Churn rate:", y_train.mean().round(2))
