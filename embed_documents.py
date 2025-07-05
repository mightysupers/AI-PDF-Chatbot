import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

# Load chunks
with open("docs/chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

print(f"✅ chunks loaded: {len(all_chunks)}")

# Handle case when chunks are empty
if len(all_chunks) == 0:
    raise ValueError("❌ No chunks loaded. Please check docs/chunks.pkl file.")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store with FAISS
db = FAISS.from_documents(all_chunks, embeddings)

# Save the FAISS index
db.save_local("vectorstore")
print("✅ FAISS vector store saved successfully in 'vectorstore/' directory.")
