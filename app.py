import streamlit as st
import os
import shutil
import requests
import zipfile
import io
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Groq
from chromadb.config import Settings

# Load environment variables
load_dotenv()

st.set_page_config(page_title="HEC Chatbot Assistant", layout="wide")
st.title("🎓 HEC Virtual Assistant")

# Define paths
DATA_DIR = "./Data/data"

# ==================== Download and unzip dataset ====================
if not os.path.exists(DATA_DIR):
    st.info("📦 Downloading HEC documents...")
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://github.com/mqasim1/HEC-chatbot-assistant/raw/main/Data.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("./Data")
    st.success("✅ Documents downloaded and extracted!")

# ==================== Load Documents ====================
@st.cache_data(show_spinner=True)
def load_documents():
    loaders = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".pdf"):
            loaders.append(PyPDFLoader(filepath))
        elif filename.endswith(".docx"):
            loaders.append(Docx2txtLoader(filepath))
    
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

# ==================== Split Documents ====================
@st.cache_data(show_spinner=True)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ==================== Embeddings ====================
@st.cache_resource(show_spinner=True)
def create_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

# ==================== Vector Store (Chroma In-Memory) ====================
@st.cache_resource(show_spinner=True)
def create_vectorstore(chunks, embeddings):
    chroma_settings = Settings(anonymized_telemetry=False)
    return Chroma.from_documents(documents=chunks, embedding=embeddings, client_settings=chroma_settings)

# ==================== Initialize Chatbot ====================
documents = load_documents()
st.success(f"✅ Loaded {len(documents)} documents")

chunks = split_documents(documents)
embeddings = create_embeddings()
db = create_vectorstore(chunks, embeddings)

retriever = db.as_retriever()

llm = Groq(model="llama3-70b-8192")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ==================== Chat Interface ====================
query = st.text_input("Ask a question about HEC:")
if query:
    with st.spinner("🔍 Searching..."):
        result = qa.run(query)
        st.success("💬 Response:")
        st.write(result)
