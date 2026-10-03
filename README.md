\# AI-Based Financial Decision System



\## Develop a Responsible MLOps Pipeline for Credit Risk Prediction



\### 1. Project Overview



This project develops a responsible Machine Learning Operations (MLOps) pipeline for credit risk prediction. The system uses customer financial information to predict credit risk and provides a risk level and decision recommendation.



The project implements machine learning, experiment tracking, model validation, model serving, monitoring, fairness analysis, audit logging, security considerations, failure handling, and human approval mechanisms.



\### 2. Dataset



The project uses the UCI German Credit Dataset.



\- Number of records: 1000

\- Input features: 20

\- Target: Credit Risk

\- 0 = Good Credit Risk

\- 1 = Bad Credit Risk



\### 3. Machine Learning Models



Three machine learning algorithms were implemented:



1\. Logistic Regression

2\. Random Forest

3\. XGBoost



\### 4. Evaluation Metrics



The models were evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- ROC-AUC



\### 5. Model Results



| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| Logistic Regression | 0.7800 | 0.6667 | 0.5333 | 0.5926 | 0.8040 |

| Random Forest | 0.7600 | 0.6500 | 0.4333 | 0.5200 | 0.7910 |

| XGBoost | 0.7550 | 0.6122 | 0.5000 | 0.5505 | 0.7867 |



\### 6. MLOps Components



The project implements:



\- Data preprocessing

\- Model training

\- Model validation

\- MLflow experiment tracking

\- Automated ML pipeline

\- FastAPI model serving

\- System monitoring

\- Fairness analysis

\- Audit trail

\- Human approval mechanism

\- Failure and edge-case testing



\### 7. Model Validation



The following validation thresholds were used:



\- Accuracy >= 0.70

\- F1 Score >= 0.50

\- ROC-AUC >= 0.75



All three trained models passed the validation criteria.



\### 8. Risk Decision Mechanism



The deployed Logistic Regression model produces a risk probability.



\- Probability < 0.40: LOW\_RISK and AUTOMATED\_DECISION

\- Probability 0.40–0.69: MEDIUM\_RISK and HUMAN\_REVIEW\_REQUIRED

\- Probability >= 0.70: HIGH\_RISK and HUMAN\_REVIEW\_REQUIRED



This mechanism prevents medium- and high-risk cases from being handled only through automated decisions.



\### 9. Model Serving



FastAPI is used to serve the trained model.



Available endpoints:



\- GET /

\- GET /health

\- POST /predict



Swagger API documentation is available at:



http://127.0.0.1:8000/docs



\### 10. Experiment Tracking



MLflow is used to track:



\- Model name

\- Dataset

\- Model metrics

\- Training configuration

\- Trained model artifacts



\### 11. Automated Pipeline



The automated pipeline is implemented in:



src/pipeline.py



The pipeline performs:



1\. Data preprocessing

2\. Model training

3\. Model validation

4\. MLflow experiment tracking



\### 12. Monitoring



The monitoring component is implemented in:



monitoring/monitor.py



It monitors:



\- Prediction count

\- Prediction distribution

\- Risk distribution

\- API latency

\- High-risk predictions



\### 13. Fairness Analysis



Group-level analysis is performed using:



src/fairness\_analysis.py



The analysis examines model performance across age groups and personal-status/sex groups using accuracy, recall, and positive prediction rate.



\### 14. Audit Trail



Predictions are recorded in JSONL format with:



\- Timestamp

\- Request ID

\- Model

\- Prediction

\- Risk probability

\- Risk level

\- Decision

\- Processing time



\### 15. Failure and Edge-Case Handling



Two failure scenarios were tested:



1\. Missing required input fields

2\. Invalid input data type



Both were correctly rejected by the API with HTTP 422 validation responses.



The API also contains handling for model unavailability and prediction errors.



\### 16. Security and Responsible MLOps



Security and responsible-AI considerations include:



\- Input validation using Pydantic

\- API error handling

\- Audit logging

\- Human review for medium/high-risk cases

\- Fairness analysis

\- Exclusion of sensitive runtime files from Git using .gitignore

\- Controlled model artifact storage



\### 17. Project Structure



```text

credit-risk-mlops-exp8/

│

├── config/

├── data/

├── dags/

├── logs/

├── models/

├── monitoring/

├── notebooks/

├── src/

│   ├── app.py

│   ├── fairness\_analysis.py

│   ├── mlflow\_tracking.py

│   ├── pipeline.py

│   ├── preprocess.py

│   ├── train.py

│   └── validate\_model.py

│

├── tests/

│   └── test\_failure\_cases.py

│

├── requirements.txt

├── model\_results.csv

├── fairness\_results.txt

└── README.md

