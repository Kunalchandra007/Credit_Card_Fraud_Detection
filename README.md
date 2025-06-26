# 💳 Credit Card Fraud Detection using Machine Learning & Deep Learning

This project is a full pipeline for detecting fraudulent credit card transactions using both supervised and unsupervised learning techniques. It includes exploratory analysis, modeling, evaluation, and a Streamlit-powered web interface for real-time prediction.

---

## 📂 Project Structure
📦 CreditCardFraudDetection/
├── Credit_Card_Fraud_Detection.ipynb # Main notebook
├── app.py # Streamlit app
├── model.pkl / logreg_model.pkl # Saved models
├── fraud_dl_model.keras # Deep learning model
├── scaler.pkl, scaler_logreg.pkl, ... # Scalers used for prediction
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 📊 Dataset Info

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraudulent Transactions**: ~0.17%
- **Features**:
  - `Time`, `Amount`
  - `V1` to `V28`: PCA-anonymized features
  - `Class`: 1 = Fraud, 0 = Legitimate

---

## 🚀 Key Features

- 📈 Data exploration with visualizations
- 🔄 Preprocessing:
  - Feature Scaling (`StandardScaler`)
  - Handling imbalance using `SMOTE`
- 🤖 Models Implemented:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - DBSCAN (unsupervised)
- 🧠 Deep Learning with Keras Sequential Model
- 📊 Evaluation Metrics:
  - Confusion Matrix, Classification Report
  - ROC-AUC Score
- 💾 Models saved via `joblib` and `keras`
- 🖥️ **Interactive Streamlit App (`app.py`)**

---

## 🖥️ Streamlit App Usage

### Features:
- Input `Time`, `Amount`, `V1–V28` manually
- Choose prediction model:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Deep Learning (Keras)
  - DBSCAN (Unsupervised)
- Output: **Fraudulent** 🚨 or **Legitimate** ✅

### Run Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```
🧠 Training & Model Saving
Trained models and scalers are saved using:

python
```bash
import joblib
joblib.dump(model, 'model_name.pkl')
model = joblib.load('model_name.pkl')
Deep learning model:
```
python
```bash
model.save("fraud_dl_model.keras")
loaded_model = tf.keras.models.load_model("fraud_dl_model.keras")
```
📈 Evaluation Metrics
Accuracy

Precision, Recall, F1-Score

ROC-AUC Score

Confusion Matrix

Focused on maximizing recall to minimize false negatives (missed frauds).

📦 Installation
```bash
git clone https://github.com/yourusername/CreditCardFraudDetection.git
cd CreditCardFraudDetection
pip install -r requirements.txt
```

📄 requirements.txt
Typical packages (to be listed in your requirements.txt):
``` bash
nginx
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn
tensorflow
keras
streamlit
joblib
Generate it with:
pip freeze > requirements.txt
```
📌 Results & Insights
Models trained on SMOTE-balanced data perform better at fraud detection.

Random Forest and XGBoost performed robustly.

Deep learning gave strong recall but needs careful tuning.

DBSCAN successfully flagged outliers (unsupervised fraud detection).


