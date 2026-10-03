import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_LOG = BASE_DIR / "logs" / "prediction_audit.jsonl"

print("=" * 60)
print("CREDIT RISK MODEL MONITORING")
print("=" * 60)

if not AUDIT_LOG.exists():
    print("No prediction logs found.")
    exit()

records = []

with open(AUDIT_LOG, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

if not records:
    print("No prediction records available.")
    exit()

risk_levels = Counter(
    record["risk_level"]
    for record in records
)

predictions = Counter(
    record["prediction"]
    for record in records
)

processing_times = [
    record["processing_time_seconds"]
    for record in records
]

total_predictions = len(records)
average_time = sum(processing_times) / total_predictions
maximum_time = max(processing_times)

print(f"\nTotal predictions : {total_predictions}")
print(f"Average latency   : {average_time:.4f} seconds")
print(f"Maximum latency   : {maximum_time:.4f} seconds")

print("\nPrediction distribution:")
for prediction, count in predictions.items():
    print(f"Prediction {prediction}: {count}")

print("\nRisk distribution:")
for risk, count in risk_levels.items():
    print(f"{risk}: {count}")

print("\nMonitoring status:")

if maximum_time > 2:
    print("WARNING: High API latency detected.")
else:
    print("API latency: NORMAL")

high_risk_count = risk_levels.get("HIGH_RISK", 0)

if high_risk_count > 0:
    print(f"WARNING: {high_risk_count} high-risk prediction(s) require human review.")
else:
    print("High-risk predictions: NONE")

print("\nMonitoring completed successfully.")
print("=" * 60)