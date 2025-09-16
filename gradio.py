import gradio as gr
import pandas as pd
import joblib
import shap

# Load your trained model
model = joblib.load("rf_tuned.pkl")

# Load dataset
data = pd.read_csv("syriatel_cleaned.csv")

# Drop target column if present
if "churn" in data.columns:
    data = data.drop(columns=["churn"])

# Encode categorical variables consistently with training
if "International_Plan" in data.columns:
    data["International_Plan"] = data["International_Plan"].map({"Yes": 1, "No": 0})
if "Voice_Mail_Plan" in data.columns:
    data["Voice_Mail_Plan"] = data["Voice_Mail_Plan"].map({"Yes": 1, "No": 0})
if "Customer_Service_Category" in data.columns:
    cust_map = {"Low": 0, "Medium": 1, "High": 2}
    data["Customer_Service_Category"] = data["Customer_Service_Category"].map(cust_map)

# Ensure numeric dtype
data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

# SHAP background sample
background = data.sample(100, random_state=42)

# SHAP explainer
explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")

# Prediction + SHAP explanation
def predict_and_explain(message, history):
    try:
        # Parse input (format: feature=value, feature=value, ...)
        data_dict = {}
        for item in message.split(","):
            key, value = item.strip().split("=")
            data_dict[key.strip()] = value.strip()

        # Convert to dataframe
        row = pd.DataFrame([data_dict])

        # Apply same encoding
        if "International_Plan" in row.columns:
            row["International_Plan"] = row["International_Plan"].map({"Yes": 1, "No": 0})
        if "Voice_Mail_Plan" in row.columns:
            row["Voice_Mail_Plan"] = row["Voice_Mail_Plan"].map({"Yes": 1, "No": 0})
        if "Customer_Service_Category" in row.columns:
            cust_map = {"Low": 0, "Medium": 1, "High": 2}
            row["Customer_Service_Category"] = row["Customer_Service_Category"].map(cust_map)

        # Convert all to numeric
        row = row.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Prediction
        prob = model.predict_proba(row)[0][1]
        pred = model.predict(row)[0]

        # SHAP explanation
        shap_values = explainer.shap_values(row)
        shap_df = pd.DataFrame({
            "feature": row.columns,
            "shap_value": shap_values[1][0],  # churn class contributions
            "row_value": row.iloc[0].values
        })
        shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)
        top_features = shap_df.head(5).to_string(index=False)

        response = f"""
📊 Prediction: {'Churn' if pred==1 else 'Not Churn'}  
🔥 Probability: {prob:.2%}  

🔎 Top factors:
{top_features}
        """
        return response

    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nPlease enter input like:\nAccount_Length=100, Total_Charges=120, Customer_Service_Calls=3, Voice_Mail_Plan=No, International_Plan=Yes"

# Launch Gradio chatbot
chatbot = gr.ChatInterface(
    fn=predict_and_explain,
    title="📞 Customer Churn Assistant",
    description="Ask me to predict churn! Provide customer details in the format: Account_Length=100, Total_Charges=120, Customer_Service_Calls=3, Voice_Mail_Plan=No, International_Plan=Yes"
)

if __name__ == "__main__":
    chatbot.launch()
