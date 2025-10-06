# -------------------------------
# app.py
# -------------------------------

from flask import Flask, render_template, request, jsonify
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# ---------------------------
# Load FAISS index + knowledge base
# ---------------------------
index = faiss.read_index("churn_index.faiss")

with open("knowledge.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# Load embedding model (same as in build_index.py)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load a Hugging Face LLM for answering
# Flan-T5 is small and runs on CPU
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# ---------------------------
# Initialize Flask app
# ---------------------------
app = Flask(__name__)

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
    user_q = request.form["msg"]

    # 1. Embed user question
    q_vec = embedder.encode([user_q])
    q_vec = np.array(q_vec).astype("float32")

    # 2. Search FAISS for most relevant knowledge
    D, I = index.search(q_vec, k=3)  # return top-3 results
    retrieved_context = " ".join([knowledge_base[i] for i in I[0]])

    # 3. Send to LLM with retrieved context
    prompt = f"Context: {retrieved_context}\n\nQuestion: {user_q}\nAnswer:"
    answer = qa_model(prompt, max_length=128, do_sample=False)[0]["generated_text"]

    return jsonify({"response": answer})

# ---------------------------
# Run Flask app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
