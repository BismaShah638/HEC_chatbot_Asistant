import os
import zipfile
import requests
import streamlit as st
from langchain.groq import Groq as GroqLLM
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# Set page config
st.set_page_config(
    page_title="HEC Chatbot Assistant",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 HEC Chatbot Assistant")

# Download and extract Data.zip from Hugging Face if not present
if not os.path.exists("./Data"):
    with st.spinner("📦 Downloading HEC Data..."):
        url = "https://huggingface.co/datasets/wasiq123/HEC/resolve/main/Data.zip"
        response = requests.get(url)
        with open("Data.zip", "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile("Data.zip", 'r') as zip_ref:
            zip_ref.extractall("./")
        os.remove("Data.zip")
        st.success("✅ Data extracted successfully!")

# Load documents
loaders = []
for root, _, files in os.walk("./Data"):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".pdf"):
            loaders.append(PyPDFLoader(file_path))
        elif file.endswith(".docx"):
            loaders.append(Docx2txtLoader(file_path))

docs = []
for loader in loaders:
    docs.extend(loader.load())

st.success(f"✅ Loaded {len(docs)} documents")

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Embeddings and Vector Store
embedding = OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index", embeddings=embedding)
else:
    db = FAISS.from_documents(documents, embedding)
    db.save_local("faiss_index")

# Set up LLM
llm = GroqLLM(temperature=0, model_name="llama3-70b-8192")
chain = load_qa_chain(llm, chain_type="stuff")

# Chat history sidebar
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("🗂️ Chat History")
    for i, entry in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {entry['question']}"):
            st.write(entry['answer'])

# Input and response area
query = st.text_input("Ask a question about HEC policies, degrees, or scholarships:")
if query:
    with st.spinner("🤖 Generating answer..."):
        docs = db.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.write("### 🧠 Answer:")
        st.write(answer)

        # Save chat
        st.session_state.chat_history.append({"question": query, "answer": answer})
