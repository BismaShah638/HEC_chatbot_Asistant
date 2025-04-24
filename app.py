import streamlit as st

# === Set Streamlit Page Config First ===
st.set_page_config(
    page_title="HEC Assistant",
    page_icon="logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import os
import time
import requests
import zipfile
import io
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit.components.v1 import html

# === CONFIG ===
ZIP_URL = "https://github.com/BismaShah638/HEC_chatbot_Asistant/raw/main/Data.zip"
DATA_PATH = "./Data"
CHROMA_PATH = "./chroma_db"

# === Download and Extract ZIP ===
def download_and_extract_zip():
    if not os.path.exists(DATA_PATH):
        st.info("📥 Downloading Data.zip from GitHub...")
        try:
            r = requests.get(ZIP_URL)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(DATA_PATH)
            st.success("✅ Data folder extracted successfully.")
        except Exception as e:
            st.error(f"❌ Failed to download or extract Data.zip: {e}")

download_and_extract_zip()

# === List Files for Debugging ===
if os.path.exists(DATA_PATH):
    st.write("📁 Files in Data folder:", os.listdir(DATA_PATH))

# === Session State Initialization ===
for key in ["messages", "conversations", "current_chat", "chat_titles", "conversation_memory"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "chat" in key or "conversation" in key else []

# === Secrets and Groq Client ===
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# === Load Documents ===
def load_documents():
    documents = []
    if not os.path.exists(DATA_PATH):
        st.warning("⚠️ 'Data' folder not found.")
        return documents

    for file in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, file)
        if file.endswith(".pdf"):
            st.write(f"📄 Loading PDF: {file}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            st.write(f"📄 Loading DOCX: {file}")
            loader = Docx2txtLoader(filepath)
            documents.extend(loader.load())

    st.success(f"✅ Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# === Load or Initialize Chroma DB ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")
if not os.path.exists(CHROMA_PATH):
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
        db.persist()
    else:
        st.error("❌ No documents found to initialize Chroma DB.")
        st.stop()
else:
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# === Sidebar Chat History ===
with st.sidebar:
    st.title("Chat History")

    if st.button("+ New Chat", use_container_width=True):
        chat_id = str(int(time.time()))
        st.session_state["current_chat"] = chat_id
        st.session_state["messages"] = []
        st.session_state["conversations"][chat_id] = []
        st.session_state["chat_titles"][chat_id] = "New Chat"
        st.session_state["conversation_memory"][chat_id] = []
        st.rerun()

    st.divider()

    for chat_id in reversed(list(st.session_state["conversations"].keys())):
        chat_title = st.session_state["chat_titles"].get(chat_id, "New Chat")
        if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state["current_chat"] = chat_id
            st.session_state["messages"] = st.session_state["conversations"][chat_id]
            st.rerun()

    st.divider()
    html("""
    <div style="margin-top: 30px;">
    <elevenlabs-convai agent-id="uYPNss1TW5NZdW1j6m5d"></elevenlabs-convai>
    <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
    </div>
    """, height=375)

# === Main Chat UI ===
st.image("logo-wide.png", use_container_width="auto")
st.title("Higher Education Commission Assistant")
st.write("Welcome to the HEC Assistant. How may I assist you with information about higher education policies, programs, or services?")

def get_groq_response(query, context, chat_memory):
    conversation_context = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_memory[-5:]
    ) if chat_memory else ""

    prompt = f"""You are a professional virtual assistant for the Higher Education Commission (HEC), Pakistan.
    Conversation context: {conversation_context}
    Context: {context}
    Question: {query}
    Answer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        stream=True,
    )
    return response

# === Show Previous Messages ===
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# === Chat Input ===
user_query = st.chat_input("Your query from HEC Assistant:")
if user_query:
    if st.session_state["current_chat"] is None:
        chat_id = str(int(time.time()))
        st.session_state["current_chat"] = chat_id
        st.session_state["conversations"][chat_id] = []
        st.session_state["chat_titles"][chat_id] = user_query[:30] + "..." if len(user_query) > 30 else user_query
        st.session_state["conversation_memory"][chat_id] = []

    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    chat_id = st.session_state["current_chat"]
    chat_memory = st.session_state["conversation_memory"].get(chat_id, [])
    st.session_state["conversation_memory"][chat_id].append({"role": "user", "content": user_query})

    results = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in get_groq_response(user_query, context, chat_memory):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
        response_placeholder.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    st.session_state["conversation_memory"][chat_id].append({"role": "assistant", "content": full_response})
    st.session_state["conversations"][chat_id] = st.session_state["messages"]
