import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "model_results.csv"

# Performance thresholds for model validation
MIN_ACCURACY = 0.70
MIN_F1 = 0.50
MIN_ROC_AUC = 0.75

results = pd.read_csv(RESULTS_FILE)

print("MODEL VALIDATION")
print("=" * 50)

all_passed = True

for _, row in results.iterrows():

    model_name = row["Model"]

    accuracy = row["Accuracy"]
    f1 = row["F1"]
    roc_auc = row["ROC_AUC"]

    accuracy_pass = accuracy >= MIN_ACCURACY
    f1_pass = f1 >= MIN_F1
    roc_auc_pass = roc_auc >= MIN_ROC_AUC

    model_passed = accuracy_pass and f1_pass and roc_auc_pass

    print(f"\nModel: {model_name}")
    print(f"Accuracy : {accuracy:.4f} -> {'PASS' if accuracy_pass else 'FAIL'}")
    print(f"F1 Score : {f1:.4f} -> {'PASS' if f1_pass else 'FAIL'}")
    print(f"ROC-AUC  : {roc_auc:.4f} -> {'PASS' if roc_auc_pass else 'FAIL'}")

    if model_passed:
        print("Validation Status: PASS")
    else:
        print("Validation Status: FAIL")
        all_passed = False

print("\n" + "=" * 50)

if all_passed:
    print("All models passed validation.")
else:
    print("One or more models failed validation.")

print("=" * 50)