import streamlit as st
import pickle
#import shap
import numpy as np
import pandas as pd

# -----------------------
# Load trained churn model
# -----------------------
with open("rf_tuned.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------
# Load background data for SHAP (numeric only, aligned with training features)
# -----------------------
background = pd.read_csv("syriatel_cleaned.csv")
background = background.drop(columns=["Churn"], errors="ignore")  # drop target if present
background = background.select_dtypes(include=[np.number]).sample(100, random_state=42)

# -----------------------
# Define chatbot questions (covering ALL model features)
# -----------------------
questions = [
    ("Account_Length", "How long has the customer been with us? (in days)"),
    ("Area_Code", "What is the customer's area code?"),
    ("Phone_Number", "Enter the last 4 digits of the phone number (just numeric placeholder):"),
    ("International_Plan", "Do they have an international plan? (yes=1, no=0)"),
    ("Voice_Mail_Plan", "Do they have a voice mail plan? (yes=1, no=0)"),
    ("Voice_Mail_Messages", "How many voice mail messages do they have?"),
    ("Total_Day_Minutes", "How many daytime minutes do they use monthly?"),
    ("Total_Day_Calls", "How many daytime calls do they make monthly?"),
    ("Total_Day_Charge", "What is their total day charge?"),
    ("Total_Evening_Minutes", "How many evening minutes do they use monthly?"),
    ("Total_Evening_Calls", "How many evening calls do they make monthly?"),
    ("Total_Evening_Charge", "What is their total evening charge?"),
    ("Total_Night_Minutes", "How many night minutes do they use monthly?"),
    ("Total_Night_Calls", "How many night calls do they make monthly?"),
    ("Total_Night_Charge", "What is their total night charge?"),
    ("Total_International_Minutes", "How many international minutes do they use monthly?"),
    ("Total_International_Calls", "How many international calls do they make monthly?"),
    ("Total_International_Charge", "What is their total international charge?"),
    ("Customer_Service_Calls", "How many times have they called customer service?"),
    ("Total_Minutes", "Enter the customer's total minutes:"),
    ("Total_Calls", "Enter the customer's total calls:"),
    ("Total_Charges", "Enter the customer's total charges:"),
    ("Avg_Call_Duration", "What is their average call duration?"),
    ("Customer_Service_Category", "Customer service category? (Low=0, Medium=1, High=2)")
]

# -----------------------
# Session state to track conversation
# -----------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

st.title("📞 AI-Powered Customer Retention Chatbot")

# -----------------------
# Chatbot flow
# -----------------------
if st.session_state.step < len(questions):
    feature, q_text = questions[st.session_state.step]
    st.chat_message("assistant").write(q_text)

    user_input = st.chat_input("Your answer...")
    if user_input:
        st.session_state.answers[feature] = user_input
        st.session_state.step += 1
        st.rerun()

else:
    # -----------------------
    # All answers collected → run prediction
    # -----------------------
    st.chat_message("assistant").write("✅ Thanks! Let me analyze the churn risk...")

    # Convert answers to dataframe
    df = pd.DataFrame([st.session_state.answers])

    # Handle categorical conversions
    df = df.replace({
        "yes": 1, "no": 0, "Yes": 1, "No": 0,
        "Low": 0, "Medium": 1, "High": 2
    })

    # Force numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure feature order matches training
    feature_order = [
        'Account_Length', 'Area_Code', 'Phone_Number', 'International_Plan',
        'Voice_Mail_Plan', 'Voice_Mail_Messages', 'Total_Day_Minutes',
        'Total_Day_Calls', 'Total_Day_Charge', 'Total_Evening_Minutes',
        'Total_Evening_Calls', 'Total_Evening_Charge', 'Total_Night_Minutes',
        'Total_Night_Calls', 'Total_Night_Charge',
        'Total_International_Minutes', 'Total_International_Calls',
        'Total_International_Charge', 'Customer_Service_Calls',
        'Total_Minutes', 'Total_Calls', 'Total_Charges',
        'Avg_Call_Duration', 'Customer_Service_Category'
    ]
    df = df.reindex(columns=feature_order)

    # Prediction
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    result = "⚠️ High Risk of Churn" if pred == 1 else "✅ Low Risk of Churn"
    st.chat_message("assistant").write(f"Prediction: **{result}** (Churn probability: {prob:.2f})")

    # -----------------------
    # SHAP explanation
    # -----------------------
    # explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    # shap_values = explainer.shap_values(df, check_additivity=False)

    # shap_df = pd.DataFrame({
        # "feature": df.columns,
        # "value": df.iloc[0].values,
        # "shap": shap_values[0]
    # }).sort_values(by="shap", key=lambda x: np.abs(x), ascending=False)

    st.write("### 🔍 Top factors influencing this prediction")
    # st.dataframe(shap_df.head(5))
