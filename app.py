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

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------
# RESPONSIVE CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    padding-top: 30px;
}

.big-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #9ca3af;
    margin-bottom: 25px;
}

.center-img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 180px;
    margin-bottom: 20px;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #2d2f36;
}

.response-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
    font-size: 16px;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .big-title {
        font-size: 28px;
    }
    .subtitle {
        font-size: 14px;
    }
    .center-img {
        width: 140px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD SECRETS
# ---------------------------------------------------
groq_api_key = st.secrets["groq_api_key"]
google_api_key = st.secrets["google_api_key"]
os.environ["GOOGLE_API_KEY"] = google_api_key

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown('<div class="big-title">DocMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your Intelligent Document Assistant</div>', unsafe_allow_html=True)

st.markdown(
    '<img src="intelligent-agent.png" class="center-img">',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📂 Upload PDF file(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("⚡ Process Document"):
        with st.spinner("Processing document..."):
            docs = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    loader = PyPDFLoader(tmp.name)
                    docs.extend(loader.load())

            if docs:
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

                st.success("✅ Document Ready!")
            else:
                st.error("No readable content found.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# LLM SETUP
# ---------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b"
)

prompt = ChatPromptTemplate.from_template("""
Answer strictly from the context below.
If the answer is not in the context, say "Answer not found in the document."

<context>
{context}
</context>

Question: {question}
""")

# ---------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
question = st.text_input("💬 Ask a question about your document")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# RETRIEVAL PIPELINE
# ---------------------------------------------------
if question and "vectors" in st.session_state:

    retriever = st.session_state.vectors.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    with st.spinner("🤖 Thinking..."):
        start = time.time()
        result = chain.invoke(question)
        end = time.time()

    st.markdown("""
    <div class="response-card">
    <b>🤖 AI Response</b><br><br>
    {}
    </div>
    """.format(result.content), unsafe_allow_html=True)

    st.write(f"⏱ Response time: {end - start:.2f} seconds")
