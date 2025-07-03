# 💳 Credit Card Fraud Detection

A machine learning-based web app to detect fraudulent credit card transactions in real time. Built with Python, scikit-learn, and Streamlit.

## 📊 Problem Statement

Credit card fraud is rare but costly. In this project, we use machine learning to:
- Handle highly imbalanced datasets
- Predict whether a transaction is fraudulent or legit
- Build a live prediction dashboard using Streamlit

## 📁 Project Structure

- `notebooks/` – Data exploration and model building (`SMOTE`, `Random Forest`, `Logistic Regression`)
- `fraud_app.py` – Streamlit app for live predictions
- `outputs/` – Trained model (`rf_model.pkl`)
- `README.md` – Project documentation

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib & Seaborn
- DATA- This project uses the public [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  

## 🔍 How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app
   ```bash
   streamlit run fraud_app.py
   ```
4. Upload a CSV file.
