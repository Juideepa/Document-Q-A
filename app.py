import streamlit as st
import os
import time
import tempfile

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Smart Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# LOAD SECRETS (Cloud Safe)
# ---------------------------
groq_api_key = st.secrets["groq_api_key"]
google_api_key = st.secrets["google_api_key"]

os.environ["GOOGLE_API_KEY"] = google_api_key

# ---------------------------
# UI
# ---------------------------
st.image("intelligent-agent.png", width=200)
st.title("DocMind AI - Your Intelligent Document Assistant")
st.markdown("Upload your PDF and ask questions from it instantly.")

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b"
)

# ---------------------------
# PROMPT
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
Answer strictly from the context below.
If the answer is not in the context, say "Answer not found in the document."

<context>
{context}
</context>

Question: {question}
""")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload your PDF file(s)",
    type="pdf",
    accept_multiple_files=True
)

# ---------------------------
# VECTOR BUILD FUNCTION
# ---------------------------
def build_vector_store(files):

    docs = []

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    if not docs:
        st.error("No text found in uploaded PDFs.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    st.session_state.vectors = FAISS.from_documents(
        final_docs,
        embeddings
    )

    st.success("PDF processed successfully! You can now ask questions.")

# ---------------------------
# PROCESS BUTTON
# ---------------------------
if uploaded_files:
    if st.button("Process PDF"):
        build_vector_store(uploaded_files)

# ---------------------------
# QUESTION INPUT
# ---------------------------
question = st.text_input("Ask something about your uploaded document")

# ---------------------------
# RETRIEVAL PIPELINE
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
