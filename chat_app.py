# ---------------------------
# Import libraries
# ---------------------------
from flask import Flask, render_template, request, jsonify
import pandas as pd
from transformers import pipeline

# ---------------------------
# Initialize Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load churn dataset
# ---------------------------
df = pd.read_csv("syriatel_cleaned.csv")

# Create a text summary of the dataset (context for Q&A model)
# Example: convert column stats into text the model can search
summary_text = ""
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    summary_text += f"The average {col} is {df[col].mean():.2f}. "
    summary_text += f"The maximum {col} is {df[col].max():.2f}. "
    summary_text += f"The minimum {col} is {df[col].min():.2f}. "

# ---------------------------
# Load Hugging Face Q&A model
# ---------------------------
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# ---------------------------
# Homepage route
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------
# Chatbot route
# ---------------------------
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["msg"]

    try:
        # Use Hugging Face Q&A pipeline
        result = qa_pipeline(question=user_input, context=summary_text)
        response = result["answer"]
    except Exception as e:
        response = f"Error: {str(e)}"

    return jsonify({"response": response})

# ---------------------------
# Run Flask app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
