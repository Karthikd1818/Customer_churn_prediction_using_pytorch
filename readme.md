# Customer Churn Prediction Using PyTorch

## Project Overview

This project uses an Artificial Neural Network (ANN) built with PyTorch to predict whether a telecom customer is likely to leave the service.

The model learns patterns from customer demographics, subscribed services, contract details, tenure, and billing information to classify customers into two categories:

- **0 — No Churn:** The customer is not predicted to leave.
- **1 — Churn:** The customer is predicted to leave.

This project demonstrates an end-to-end deep learning workflow, from data preprocessing to model training, evaluation, and prediction on new customer records.

## Objectives

- Clean and preprocess customer data.
- Encode categorical features into numerical values.
- Split data into training and testing sets.
- Scale features without data leakage.
- Build and train an ANN using PyTorch.
- Evaluate the model using classification metrics.
- Save the trained model and scaler.
- Predict churn probability for a new customer.

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains customer information, including:

- Customer tenure
- Monthly and total charges
- Contract type
- Internet and phone services
- Payment method
- Customer churn status

The `Churn` column is the target variable.

## Technologies Used

- Python
- Pandas
- Sc