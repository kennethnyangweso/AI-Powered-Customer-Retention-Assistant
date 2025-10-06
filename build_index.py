# -------------------------------
# build_index.py
# -------------------------------

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# 1. Load churn dataset
df = pd.read_csv("syriatel_cleaned.csv")

# 2. Build text knowledge base from dataset
#    → Summarize numeric columns into text form
knowledge_base = []
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        knowledge_base.append(
            f"The average {col} is {df[col].mean():.2f}, "
            f"the max is {df[col].max():.2f}, "
            f"and the min is {df[col].min():.2f}."
        )
    else:
        top_vals = df[col].value_counts().head(3).to_dict()
        knowledge_base.append(f"Column {col} has top values: {top_vals}")

# 3. Load a sentence transformer to create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert knowledge base into embeddings
embeddings = model.encode(knowledge_base, convert_to_numpy=True)

# 5. Create FAISS index
dimension = embeddings.shape[1]  # embedding size (384 for MiniLM)
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. Save FAISS index and metadata (knowledge_base text)
faiss.write_index(index, "churn_index.faiss")
with open("knowledge.pkl", "wb") as f:
    pickle.dump(knowledge_base, f)

print("✅ FAISS index and knowledge base saved!")
