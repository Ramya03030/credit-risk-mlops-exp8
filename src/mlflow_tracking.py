import mlflow
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "model_results.csv"
MODELS_DIR = BASE_DIR / "models"

# Use SQLite database for MLflow tracking
MLFLOW_DB = BASE_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

# Create MLflow experiment
mlflow.set_experiment("Credit_Risk_Model_Comparison")

# Read model comparison results
results = pd.read_csv(RESULTS_FILE)

# Model file mapping
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Log each model
for _, row in results.iterrows():

    model_name = row["Model"]
    model_file = model_files[model_name]

    model_path = MODELS_DIR / model_file

    # Verify model exists
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        continue

    # Load model to verify it is valid
    model = joblib.load(model_path)

    with mlflow.start_run(run_name=model_name):

        # Parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", "UCI German Credit")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        # Metrics
        mlflow.log_metric("accuracy", float(row["Accuracy"]))
        mlflow.log_metric("precision", float(row["Precision"]))
        mlflow.log_metric("recall", float(row["Recall"]))
        mlflow.log_metric("f1_score", float(row["F1"]))
        mlflow.log_metric("roc_auc", float(row["ROC_AUC"]))

        # Log the trained model file as an artifact
        mlflow.log_artifact(
            str(model_path),
            artifact_path="trained_model"
        )

        # Log model result information
        mlflow.log_text(
            f"Model: {model_name}\n"
            f"Accuracy: {row['Accuracy']}\n"
            f"Precision: {row['Precision']}\n"
            f"Recall: {row['Recall']}\n"
            f"F1 Score: {row['F1']}\n"
            f"ROC-AUC: {row['ROC_AUC']}\n",
            "model_metrics.txt"
        )

        print(f"Logged {model_name} to MLflow")

print("\nMLflow tracking completed successfully.")
print(f"MLflow database: {MLFLOW_DB}")