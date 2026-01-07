from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import *

def get_qa_chain():
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return llm, retriever

