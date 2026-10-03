import requests

API_URL = "http://127.0.0.1:8000/predict"

print("=" * 60)
print("FAILURE AND EDGE-CASE TESTING")
print("=" * 60)

# ---------------------------------------------------------
# Test 1: Missing required field
# ---------------------------------------------------------

print("\nTEST 1: Missing required field")

invalid_data = {
    "checking_status": "A11",
    "duration_months": 6,
    "credit_history": "A34"
}

response = requests.post(API_URL, json=invalid_data)

print("HTTP Status:", response.status_code)

if response.status_code == 422:
    print("PASS: Invalid input was rejected.")
else:
    print("FAIL: Invalid input was not rejected.")


# ---------------------------------------------------------
# Test 2: Invalid data type
# ---------------------------------------------------------

print("\nTEST 2: Invalid data type")

invalid_data = {
    "checking_status": "A11",
    "duration_months": "invalid",
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1169,
    "savings_status": "A65",
    "employment_since": "A75",
    "installment_rate": 4,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "residence_since": 4,
    "property": "A121",
    "age": 67,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 2,
    "job": "A173",
    "dependents": 1,
    "telephone": "A192",
    "foreign_worker": "A201"
}

response = requests.post(API_URL, json=invalid_data)

print("HTTP Status:", response.status_code)

if response.status_code == 422:
    print("PASS: Invalid data type was rejected.")
else:
    print("FAIL: Invalid data type was not rejected.")


print("\n" + "=" * 60)
print("Failure and edge-case testing completed.")
print("=" * 60)