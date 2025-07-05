# 🧠 AI PDF Chatbot using LLaMA 2 (Fully Offline)

This project allows you to chat with your own PDF documents using the LLaMA 2 model completely offline — no API keys required!  
Built with LangChain, FAISS, HuggingFace Embeddings, and llama-cpp-python.

## 🚀 Features
- Load any PDF file and convert it into chunks.
- Generate embeddings using `sentence-transformers`.
- Store & retrieve using FAISS.
- Query offline using a locally downloaded LLaMA 2 model.
- Works 100% without internet once setup.

## 📁 Folder Structure

```
AI-PDF-Chatbot/
│
├── docs/                   # Your PDFs and generated chunks
│   ├── sample.pdf
│   └── chunks.pkl
│
├── models/                 # (DO NOT upload to GitHub) contains .gguf LLaMA 2 model
│
├── load_docs.py            # Splits and chunks your PDF
├── embed_documents.py      # Generates vector embeddings
├── chat_with_llama.py      # Main chatbot script
│
├── requirements.txt        # Python dependencies
└── README.md               # You're reading it!
```

## 🛠️ Setup Instructions

```bash
# 1. Create virtual environment (optional)
python -m venv .venv
source .venv/Scripts/activate  # for Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## 🧠 Model
Download LLaMA 2 (GGUF format) from trusted sources like HuggingFace or TheBloke. Place inside `/models/`.

## 🤖 Start Chatting

```bash
python chat_with_llama.py
```

## 🙏 Acknowledgments
- Meta AI for LLaMA 2
- LangChain community
- HuggingFace transformers