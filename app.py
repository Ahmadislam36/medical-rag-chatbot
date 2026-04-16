import streamlit as st
from rag_pipeline import ask_question

st.title("🏥 Medical RAG Chatbot")

query = st.text_input("Ask your medical question:")

if query:
    with st.spinner("Thinking..."):
    
        answer, context = ask_question(query)

    st.subheader("Answer:")
    st.write(answer)

