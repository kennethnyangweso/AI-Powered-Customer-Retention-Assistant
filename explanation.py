import shap
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load model and data
# -------------------------------
model = joblib.load("rf_tuned.pkl")  # replace with your model path
data = pd.read_csv("syriatel_cleaned.csv")    # replace with your dataset path

# -------------------------------
# Ensure data is numeric
# -------------------------------
# Convert object/categorical columns to numeric (e.g., one-hot encoding or label encoding)
data = pd.get_dummies(data, drop_first=True)

# -------------------------------
# Create background sample (for SHAP)
# -------------------------------
background_sample = data.sample(100, random_state=42)  # pick 100 random rows
background_sample = background_sample.astype(np.float32).values  # convert to float

# -------------------------------
# Define SHAP explanation function
# -------------------------------
def shap_explain_single(model, row, background):
    """
    Explain a single prediction using SHAP.
    """
    # make sure row is in correct format
    row = row.astype(np.float32).values.reshape(1, -1)

    explainer = shap.TreeExplainer(
        model, 
        data=background, 
        feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(row)

    # Convert to DataFrame for better readability
    shap_df = pd.DataFrame({
        "feature": data.columns,
        "shap_value": shap_values[0],
        "row_value": row.flatten()
    }).sort_values(by="shap_value", key=abs, ascending=False)

    return shap_df

# -------------------------------
# Example usage: explain one row
# -------------------------------
row = data.iloc[0]  # pick first row
shap_df = shap_explain_single(model, row, background_sample)

print(shap_df.head(10))  # top 10 important features


