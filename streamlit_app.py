import streamlit as st

st.title("📄 AI PDF Chatbot with LLaMA2")
st.write("Ask anything from your PDF.")

query = st.text_input("You:", "")

if query:
    # Placeholder response — replace with actual llama response code if running locally
    st.write("🤖 Bot:", f"Sorry, I cannot respond here without GPU model support. You asked: {query}")
