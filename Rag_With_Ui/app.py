import streamlit as st
from main import query_rag

st.set_page_config(page_title="RAG Q&A", layout="centered")

st.title("📚 RAG-based Question Answering")

user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Fetching answer..."):
            try:
                answer = query_rag(user_query)
                st.success("✅ Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question before submitting.")