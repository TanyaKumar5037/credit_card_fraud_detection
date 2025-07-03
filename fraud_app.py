import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("outputs/rf_model.pkl")

# Streamlit page configuration
st.set_page_config(page_title="Fraud Detector", page_icon="💳")
st.title("💳 Credit Card Fraud Detection App")

st.markdown(
    "Upload a CSV file. If the 'Class' column is present, it will be ignored during prediction. "
    "The model will analyze the transactions and predict which ones are potentially fraudulent."
)

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop 'Class' column if it exists (target label should not be used in prediction)
    if 'Class' in df.columns:
        df.drop('Class', axis=1, inplace=True)

    st.subheader("Uploaded Data (first 5 rows)")
    st.write(df.head())

    # Drop 'Time' column if it exists (often not useful for modeling)
    if 'Time' in df.columns:
        df.drop('Time', axis=1, inplace=True)

    # Scale 'Amount' column if present
    if 'Amount' in df.columns:
        scaler = StandardScaler()
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Predict using the trained model
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]  # probability of being fraud

    # Add predictions and probabilities to the dataframe
    df['Prediction'] = preds
    df['Fraud Probability'] = probs

    # Display prediction summary
    st.subheader("🔍 Prediction Summary")
    st.write(df['Prediction'].value_counts().rename(index={0: 'Legit', 1: 'Fraud'}))

    # Display preview of results
    st.subheader("📊 Results Preview (First 20 Rows)")
    st.dataframe(df.head(20))

    # Download button for results
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
