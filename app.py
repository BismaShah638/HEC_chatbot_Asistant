import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Groq

# Page configuration
st.set_page_config(
    page_title="HEC Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title("🎓 HEC Virtual Assistant")

# Function to load all .pdf and .docx files from current directory
def load_documents():
    documents = []
    for file in os.listdir("."):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
        elif file.endswith(".docx"):
            try:
                loader = Docx2txtLoader(file)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
    return documents

# Text splitter
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Load vector database (FAISS)
@st.cache_resource(show_spinner="🔍 Indexing documents...")
def load_vector_store():
    raw_docs = load_documents()
    if not raw_docs:
        st.error("❌ No PDF or DOCX files found in the root directory.")
        return None
    chunks = split_documents(raw_docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(chunks, embeddings)

# Initialize retriever
vector_store = load_vector_store()
if vector_store is not None:
    retriever = vector_store.as_retriever()
else:
    st.stop()

# Initialize Groq LLM
llm = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-70b-8192"
)

# Setup QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role, text = msg
    if role == "user":
        st.chat_message("user").markdown(text)
    else:
        st.chat_message("assistant").markdown(text)

# Chat input
prompt = st.chat_input("Ask me something about HEC...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(("user", prompt))

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = qa_chain.run(prompt)
                st.markdown(response)
                st.session_state.messages.append(("assistant", response))
            except Exception as e:
                st.error(f"💥 Failed to generate response: {e}")
