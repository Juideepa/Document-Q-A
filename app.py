import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------
# Load API keys
# ---------------------------
load_dotenv()

groq_api_key = st.secrets["groq_api_key"]
google_api_key = st.secrets["google_api_key"]

os.environ["GOOGLE_API_KEY"] = google_api_key

# ---------------------------
# UI
# ---------------------------
st.image("intelligent-agent.png", width=200)
st.title("Your Smart Q&A Intelligent Assistant")

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b"
)

# ---------------------------
# Prompt template
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
Answer strictly from the context.

<context>
{context}
</context>

Question: {question}
""")

# ---------------------------
# Vector embedding
# ---------------------------
def vector_embedding():

    if "vectors" in st.session_state:
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    loader = PyPDFDirectoryLoader("./ed_pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_docs = splitter.split_documents(docs)

    st.session_state.vectors = FAISS.from_documents(
        final_docs,
        embeddings
    )

# ---------------------------
# Input
# ---------------------------
question = st.text_input("Ask something about the document")

if st.button("Load database"):
    vector_embedding()
    st.success("Database ready!")

# ---------------------------
# Retrieval pipeline
# ---------------------------
if question and "vectors" in st.session_state:

    retriever = st.session_state.vectors.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    start = time.time()
    result = chain.invoke(question)
    end = time.time()

    st.markdown("### AI Response")
    st.success(result.content)

    st.write(f"Response time: {end - start:.2f} seconds")

