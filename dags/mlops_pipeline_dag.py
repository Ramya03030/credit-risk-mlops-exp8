"""
Credit Risk MLOps Pipeline Workflow

Workflow:
1. Data preprocessing
2. Model training
3. Model validation
4. MLflow experiment tracking

The actual execution is handled by src/pipeline.py.
This file documents the pipeline stages and their dependency order.
"""

PIPELINE_STAGES = [
    "Data Preprocessing",
    "Model Training",
    "Model Validation",
    "MLflow Experiment Tracking"
]

DEPENDENCIES = {
    "Data Preprocessing": [],
    "Model Training": ["Data Preprocessing"],
    "Model Validation": ["Model Training"],
    "MLflow Experiment Tracking": ["Model Validation"]
}


if __name__ == "__main__":
    print("Credit Risk MLOps Pipeline DAG")
    print("=" * 40)

    for stage in PIPELINE_STAGES:
        print(f"→ {stage}")

    print("\nPipeline dependency order:")
    for stage, dependencies in DEPENDENCIES.items():
        print(f"{stage} <- {dependencies}")