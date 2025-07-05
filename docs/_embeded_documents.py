# step2_embed_documents.py

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import pickle

# 🔁 Load chunked documents from step 1
with open("docs.pkl", "rb") as f:
    chunks = pickle.load(f)

# 🔤 Create OpenAI Embeddings
embedding = OpenAIEmbeddings()

# 🧠 Store embeddings in Chroma
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="chroma_db")
vectordb.persist()

print("✅ Step 2 done: Embeddings stored in Chroma DB.")
