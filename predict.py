
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn

# 1. Load the training feature columns
with open("feature_columns.json", "r") as file:
    feature_columns = json.load(file)

# 2. Recreate the same neural network architecture
input_size = len(feature_columns)

class ChurnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# 3. Load the trained model
model = ChurnModel()
model.load_state_dict(
    torch.load("churn_model.pth", map_location="cpu", weights_only=True)
)
model.eval()

# 4. Load the fitted scaler
with open("churn_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# 5. Example customer data
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
}])

# 6. Encode categorical columns like training data
customer_encoded = pd.get_dummies(
    customer,
    drop_first=True,
    dtype=int
)

# Align columns and order with training data
customer_encoded = customer_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# 7. Scale the customer data
customer_scaled = scaler.transform(customer_encoded)

# 8. Convert to a PyTorch tensor
customer_tensor = torch.tensor(
    customer_scaled,
    dtype=torch.float32
)

# 9. Predict churn probability
with torch.no_grad():
    logit = model(customer_tensor)
    probability = torch.sigmoid(logit).item()

print(f"Churn probability: {probability:.2%}")

if probability >= 0.5:
    print("Prediction: Customer may churn")
else:
    print("Prediction: Customer may not churn")
