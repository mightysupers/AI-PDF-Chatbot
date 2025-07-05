from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import pickle

# 🔹 Load your FAISS index
with open("docs/faiss_index.pkl", "rb") as f:
    db = pickle.load(f)

# 🔹 Set up retriever
retriever = db.as_retriever()

# 🔹 Load local LLaMA model
llm = LlamaCpp(
    model_path="models/llama-2-7b.ggmlv3.q4_0.bin",  # 🛑 Change this path to your model file
    temperature=0.7,
    max_tokens=512,
    top_p=1,
    n_ctx=2048,
    n_batch=512,
    verbose=True
)

# 🔹 Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🔹 Chat loop
while True:
    query = input("\n🧠 Ask a question (or 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break
    result = qa_chain.run(query)
    print(f"\n🤖 Answer: {result}")
