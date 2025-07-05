# step1_load_docs.py
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Loading file: {filename}")
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            documents.extend(docs)
    return documents

docs_folder = "docs"

documents = load_documents_from_folder(docs_folder)
print(f"✅ Total documents loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")

with open("docs.pkl", "wb") as f:
    pickle.dump(chunks, f)
