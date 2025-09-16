# query_index.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --------------------------
# 1. Load embedding model
# --------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example customer churn insights (replace with your dataset)
docs = [
    "High customer service calls drive churn.",
    "Long contracts reduce churn likelihood.",
    "International plan increases churn risk.",
    "Voice mail plan reduces churn likelihood.",
    "High data usage correlates with higher churn probability."
]

# --------------------------
# 2. Build FAISS index
# --------------------------
embeddings = embedder.encode(docs, convert_to_numpy=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --------------------------
# 3. Load summarization model
# --------------------------
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# --------------------------
# 4. Query function
# --------------------------
def query_bot(question, k=3):
    # Encode question
    query_vec = embedder.encode([question], convert_to_numpy=True)

    # Search FAISS index
    D, I = index.search(query_vec, k=k)

    # Retrieve top docs
    context = " ".join([docs[i] for i in I[0]])

    # Use Hugging Face model to summarize
    prompt = f"Question: {question}\nContext: {context}\nAnswer in a concise way:"
    response = qa_pipeline(prompt, max_length=100, do_sample=False)

    return response[0]["generated_text"]

# --------------------------
# 5. Test query
# --------------------------
if __name__ == "__main__":
    print(query_bot("What features drive customer churn the most?"))
    print(query_bot("How can churn be reduced?"))


