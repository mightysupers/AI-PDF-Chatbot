from langchain_community.llms import LlamaCpp
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ✅ Load LangChain-compatible LLaMA model
llm = LlamaCpp(
    model_path="models/llama-2-7b.Q8_0.gguf",
    n_ctx=2048,
    temperature=0.7,
    verbose=True
)

# ✅ Load document chunks
with open("docs/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ✅ Load FAISS index from documents
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ Define prompt template
template = """Use the following documents to answer the question.
If you don’t know the answer, just say you don’t know.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# ✅ Create RAG (Retrieval-Augmented Generation) Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# ✅ Chat loop
print("🤖 Ask anything from your PDFs (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = qa_chain.invoke(query)

    print("Bot:", response)
