import subprocess
import sys

print("=" * 60)
print("CREDIT RISK MLOPS PIPELINE")
print("=" * 60)

steps = [
    ("Data Preprocessing", "src\\preprocess.py"),
    ("Model Training", "src\\train.py"),
    ("Model Validation", "src\\validate_model.py"),
    ("MLflow Tracking", "src\\mlflow_tracking.py")
]

for step_name, script in steps:

    print(f"\n{'-' * 60}")
    print(f"Running: {step_name}")
    print(f"{'-' * 60}")

    result = subprocess.run(
        [sys.executable, script],
        check=False
    )

    if result.returncode != 0:
        print(f"\nPipeline failed at: {step_name}")
        sys.exit(1)

    print(f"{step_name} completed successfully.")

print("\n" + "=" * 60)
print("COMPLETE MLOPS PIPELINE EXECUTED SUCCESSFULLY")
print("=" * 60)