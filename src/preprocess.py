import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "german.data"
OUTPUT_FILE = BASE_DIR / "data" / "credit_data.csv"

# Column names from the German Credit dataset
columns = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]

# Read raw dataset
df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

# Convert target:
# 1 = Good credit risk
# 2 = Bad credit risk
df["credit_risk"] = df["credit_risk"].map({
    1: 0,
    2: 1
})

# Save clean dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Dataset preprocessing completed successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nTarget distribution:")
print(df["credit_risk"].value_counts())
print("\nFirst 5 rows:")
print(df.head())