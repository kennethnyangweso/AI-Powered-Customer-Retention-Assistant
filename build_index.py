# build_index.py
import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "syriatel_cleaned.csv"   
OUT_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(OUT_DIR, exist_ok=True)

def row_to_doc(row, idx):
    # Build a compact, human-readable summary of each row (include top features)
    parts = [
        f"CustomerID: {idx}",
        f"Account_Length: {row.get('Account_Length', '')}",
        f"Area_Code: {row.get('Area_Code', '')}",
        f"International_Plan: {row.get('International_Plan', '')}",
        f"Voice_Mail_Plan: {row.get('Voice_Mail_Plan', '')}",
        f"Voice_Mail_Messages: {row.get('Voice_Mail_Messages', '')}",
        f"Total_Day_Minutes: {row.get('Total_Day_Minutes', '')}",
        f"Total_Day_Charge: {row.get('Total_Day_Charge', '')}",
        f"Total_Minutes: {row.get('Total_Minutes', '')}",
        f"Total_Calls: {row.get('Total_Calls', '')}",
        f"Total_Charges: {row.get('Total_Charges', '')}",
        f"Customer_Service_Calls: {row.get('Customer_Service_Calls', '')}",
        f"Customer_Service_Category: {row.get('Customer_Service_Category', '')}",
        f"Total_International_Minutes: {row.get('Total_International_Minutes', '')}",
        f"Total_International_Calls: {row.get('Total_International_Calls', '')}",
        f"Churn: {row.get('Churn', '')}"
    ]
    return " | ".join([p for p in parts if p is not None and p != ""])

def main():
    print("Loading CSV:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    docs = []
    metas = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        txt = row_to_doc(row, idx)
        docs.append(txt)
        metas.append({"index": int(idx), "churn": int(row.get("Churn", -1))})

    print("Loading embedder:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Encoding documents...")
    embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    # normalize for cosine (optional but improves results)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)

    dim = embeddings.shape[1]
    print("Embedding dimension:", dim)

    index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    faiss_path = os.path.join(OUT_DIR, "faiss.index")
    docs_path = os.path.join(OUT_DIR, "docs.pkl")
    print("Saving FAISS index ->", faiss_path)
    faiss.write_index(index, faiss_path)
    print("Saving docs metadata ->", docs_path)
    with open(docs_path, "wb") as f:
        pickle.dump({"docs": docs, "metas": metas, "model": EMBED_MODEL}, f)

    print("Index build complete. Documents indexed:", len(docs))

if __name__ == "__main__":
    main()

