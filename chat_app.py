# ---------------------------
# Import libraries
# ---------------------------
from flask import Flask, render_template, request, jsonify   # For Flask web app
import pandas as pd                                          # For dataset handling
from transformers import pipeline                            # For Hugging Face Q&A model

# ---------------------------
# Initialize Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load churn dataset
# ---------------------------
df = pd.read_csv("syriatel_cleaned.csv")

# Convert column names to lowercase (avoid case issues)
df.columns = [col.lower() for col in df.columns]

# ---------------------------
# Create a richer summary text (context for Q&A model)
# ---------------------------
summary_text = ""
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    summary_text += f"The average {col} is {df[col].mean():.2f}. "
    summary_text += f"The maximum {col} is {df[col].max():.2f}. "
    summary_text += f"The minimum {col} is {df[col].min():.2f}. "

# Add churn distribution
if "churn" in df.columns:
    churn_counts = df["churn"].value_counts(normalize=True) * 100
    summary_text += f"The churn distribution is: {churn_counts.to_dict()}. "

# ---------------------------
# Load Hugging Face Q&A model
# ---------------------------
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# ---------------------------
# Homepage route
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")   # Loads templates/index.html

# ---------------------------
# Chatbot route
# ---------------------------
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["msg"].lower()   # Get user question

    try:
        # 1. Overall churn rate
        if "overall churn" in user_input or "churn rate" in user_input:
            churn_rate = df["churn"].mean() * 100
            response = f"The overall churn rate is {churn_rate:.2f}%."

        # 2. Total churned customers
        elif "how many customers" in user_input and "churn" in user_input:
            total_churned = df["churn"].sum()
            response = f"A total of {total_churned} customers have churned."

        # 3. International plan effect
        elif "international plan" in user_input:
            if "international_plan" in df.columns:
                churn_by_plan = df.groupby("international_plan")["churn"].mean() * 100
                response = f"Churn rate by international plan: {churn_by_plan.to_dict()}"
            else:
                response = "The dataset does not contain an 'international_plan' column."

        # 4. State with highest churn
        elif "state" in user_input:
            if "state" in df.columns:
                churn_by_state = df.groupby("state")["churn"].mean() * 100
                top_state = churn_by_state.idxmax()
                top_rate = churn_by_state.max()
                response = f"The state with the highest churn rate is {top_state} ({top_rate:.2f}%)."
            else:
                response = "The dataset does not contain a 'state' column."

        # 5. Average charges of churned customers
        elif "average total charges" in user_input:
            if "total_charges" in df.columns:
                avg_charges_churned = df[df["churn"] == 1]["total_charges"].mean()
                response = f"The average total charges for churned customers is {avg_charges_churned:.2f}."
            else:
                response = "The dataset does not contain a 'total_charges' column."

        # 6. Fallback: Use Hugging Face Q&A
        else:
            result = qa_pipeline(question=user_input, context=summary_text)
            response = result["answer"]

    except Exception as e:
        response = f"⚠️ Error: {str(e)}"

    return jsonify({"response": str(response)})

# ---------------------------
# Run Flask app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
