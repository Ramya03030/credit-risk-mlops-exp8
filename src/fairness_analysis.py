import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "credit_data.csv"
MODEL_FILE = BASE_DIR / "models" / "logistic_regression.pkl"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Separate features and target
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# Use the same test split as model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load(MODEL_FILE)

# Generate predictions
predictions = model.predict(X_test)

# Add predictions and target to test data
analysis = X_test.copy()
analysis["actual"] = y_test.values
analysis["prediction"] = predictions

print("=" * 60)
print("BIAS AND FAIRNESS ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# AGE GROUP ANALYSIS
# ---------------------------------------------------------

def age_group(age):
    if age < 30:
        return "Young (<30)"
    elif age < 50:
        return "Middle (30-49)"
    else:
        return "Older (50+)"

analysis["age_group"] = analysis["age"].apply(age_group)

print("\nAGE GROUP PERFORMANCE")
print("-" * 60)

for group, data in analysis.groupby("age_group"):

    if len(data) == 0:
        continue

    accuracy = accuracy_score(
        data["actual"],
        data["prediction"]
    )

    recall = recall_score(
        data["actual"],
        data["prediction"],
        zero_division=0
    )

    positive_prediction_rate = data["prediction"].mean()

    print(f"\nGroup: {group}")
    print(f"Samples: {len(data)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Positive prediction rate: {positive_prediction_rate:.4f}")


# ---------------------------------------------------------
# PERSONAL STATUS / SEX ANALYSIS
# ---------------------------------------------------------

print("\nPERSONAL STATUS / SEX GROUP ANALYSIS")
print("-" * 60)

for group, data in analysis.groupby("personal_status_sex"):

    accuracy = accuracy_score(
        data["actual"],
        data["prediction"]
    )

    recall = recall_score(
        data["actual"],
        data["prediction"],
        zero_division=0
    )

    positive_prediction_rate = data["prediction"].mean()

    print(f"\nGroup: {group}")
    print(f"Samples: {len(data)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Positive prediction rate: {positive_prediction_rate:.4f}")


print("\n" + "=" * 60)
print("Fairness analysis completed successfully.")
print("Group-level metrics should be reviewed before production use.")
print("=" * 60)