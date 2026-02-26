# 🏠 Real Estate Investment Advisor  
### Predicting Property Profitability & Future Value using Machine Learning

An **end-to-end Machine Learning & MLOps project** that helps investors evaluate real estate properties by predicting **future price growth** and identifying **good investment opportunities** using data-driven insights.

---

## 🚀 Project Overview

This project builds a **Real Estate Investment Advisor** that:

- 📈 Predicts **future property price (5 years ahead)**
- ✅ Classifies properties as **Good / Risky Investment**
- 🧠 Uses **advanced feature engineering & ML models**
- 🧪 Tracks experiments using **MLflow**
- 🖥️ Provides an interactive **Streamlit web application**

The system is designed to mimic a real-world real estate analytics platform.

---

## 🧩 Problem Statements

### 🔹 1. Investment Classification
**Goal:** Determine whether a property is a **Good Investment**

**Target Variable:** `Good_Investment` (Binary)

Decision is based on:
- Property pricing
- Amenities availability
- Transport connectivity
- Property age
- Location intelligence

---

### 🔹 2. Future Price Prediction
**Goal:** Predict the **future price after 5 years**

**Target Variable:** `Future_Price_5Y` (Regression)

Predictions are learned from:
- Property features
- City & locality trends
- Amenities and infrastructure
- Historical pricing patterns

---

## 📊 Dataset

- **Type:** Synthetic Indian real estate dataset
- **Size:** ~250,000 records
- **Key Features:**
  - BHK
  - Size (SqFt)
  - Price per SqFt
  - Property Type
  - City & Locality
  - Amenities
  - Furnishing Status
  - Transport Connectivity
  - Property Age

> ⚠️ Large datasets are excluded from GitHub.  
> Refer to preprocessing steps in the notebook.

---

## 🛠️ Feature Engineering

- **Amenities Engineering**
  - Binary flags (`Has_Gym`, `Has_Pool`, etc.)
  - `Amenity_Count`

- **Location Intelligence**
  - City Price Score
  - Locality Price Score

- **Transport Score**
  - Low / Medium / High → 1 / 2 / 3

- **Investment Scoring**
  - Multi-factor logic used to generate classification labels

---

## 🤖 Machine Learning Models

### 🔹 Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- **XGBoost Classifier**

**Evaluation Metrics**
- Accuracy
- Precision
- Recall
- ROC AUC

---

### 🔹 Regression Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- **XGBoost Regressor**

**Evaluation Metrics**
- RMSE
- MAE
- R² Score

---

## 🧪 MLflow Experiment Tracking

- Used **MLflow** for experiment management
- Logged:
  - Model parameters
  - Evaluation metrics
  - Trained models
- Registered best models:
  - `XGBoost_Classifier`
  - `XGBoost_Regressor`

MLflow artifacts are stored **outside notebooks** for clean project structure.

---

## 🖥️ Streamlit Application

### 🔹 Features
- Multi-page interface
- Real-time investment prediction
- Future price estimation
- Data insights & visualizations
- Feature importance analysis
- Creator information page

### 🔹 Inputs
- BHK & property size
- Amenities selection
- Furnishing status
- Transport connectivity
- City & locality scores

### 🔹 Outputs
- ✅ Good / Risky Investment decision
- 📊 Confidence score
- 💰 Predicted future price (5 years)

---


