import streamlit as st
import os
from dotenv import load_dotenv

from loaders.url_loader import load_urls
from embeddings.embedder import build_vectorstore
from chains.qa_chain import get_qa_chain

load_dotenv()

st.title("DSA-GPT 🚀")
st.subheader("Your AI Study Buddy for Coding Interviews")

urls = [st.text_input(f"Article URL {i+1}") for i in range(3)]


if st.button("Process URLs"):
    try:
        if not any(urls):
            st.warning("Please enter at least one URL")
        else:
            with st.spinner("Processing URLs..."):
                docs = load_urls(urls)
                build_vectorstore(docs)
            st.success("Knowledge Base Created!")
    except Exception as e:
        st.error("URL processing failed.")
        st.code(str(e))


query = st.text_input("Ask your question")

if query:
    if not os.path.exists("data/faiss_index"):
        st.warning("Please process URLs first.")
    else:
        llm, retriever = get_qa_chain()

        docs = retriever.invoke(query)


        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a DSA interview mentor.

Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know based on the given sources."

Context:
{context}

Question:
{query}
"""

        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)


        st.header("Answer")
        st.write(response.content)

        if docs:
            st.subheader("Sources")
            shown = set()
            for doc in docs:
                src = doc.metadata.get("source", "")
                if src and src not in shown:
                    st.write(src)
                    shown.add(src)
