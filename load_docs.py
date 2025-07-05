import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading file: {filename}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ {len(docs)} pages loaded from {filename}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ✅ Start of script
docs_folder = "docs"
if not os.path.exists(docs_folder):
    print("❌ 'docs' folder not found. Please make sure it exists.")
    exit()

documents = load_documents_from_folder(docs_folder)
if not documents:
    print("❌ No PDF documents found in the 'docs' folder.")
    exit()

chunks = split_documents(documents)

print(f"\n📚 Total documents loaded: {len(documents)}")
print(f"🔖 Total chunks created: {len(chunks)}")

# ✅ Save to chunks.pkl
os.makedirs("docs", exist_ok=True)
with open("docs/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ chunks.pkl saved successfully.")
