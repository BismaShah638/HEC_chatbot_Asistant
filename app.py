import streamlit as st
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

# === Auto-download and extract ZIP if Data folder is missing ===
def download_and_extract_zip_from_github():
    data_path = "./Data"
    if not os.path.exists(data_path):
        st.info("📥 Downloading Data.zip from GitHub...")
        try:
            r = requests.get(ZIP_URL)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(data_path)
            st.success("✅ Data folder extracted successfully.")
        except Exception as e:
            st.error(f"❌ Failed to download or extract Data.zip: {e}")

download_and_extract_zip_from_github()

# === Session state initialization ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = {}

# === Secrets and Groq client ===
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# === Load documents recursively ===
def load_documents():
    documents = []
    data_path = "./Data"

    if not os.path.exists(data_path):
        st.warning("⚠️ 'Data' folder not found. Skipping document loading.")
        return documents

    file_count = 0

    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                st.write(f"📄 Loading PDF: {file}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                file_count += 1
            elif file.endswith(".docx"):
                st.write(f"📄 Loading DOCX: {file}")
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                file_count += 1

    if file_count == 0:
        st.warning("⚠️ No PDF or DOCX files found in the 'Data' folder.")
    else:
        st.success(f"✅ Loaded {file_count} files with {len(documents)} document chunks.")

    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    documents = load_documents()
    if documents:
        chunks = split_documents(documents)
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        db.persist()
    else:
        st.error("❌ No documents found to initialize Chroma DB.")
        st.stop()
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# === Sidebar Chat History ===
with st.sidebar:
    st.title("Chat History")

    if st.button("+ New Chat", use_container_width=True):
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.messages = []
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = "New Chat"
        st.session_state.conversation_memory[chat_id] = []
        st.rerun()

    st.divider()

    for chat_id in reversed(list(st.session_state.conversations.keys())):
        chat_title = st.session_state.chat_titles.get(chat_id, "New Chat")
        if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat = chat_id
            st.session_state.messages = st.session_state.conversations[chat_id]
            st.rerun()

    st.divider()
    html("""
    <div style="margin-top: 30px;">
    <elevenlabs-convai agent-id=\"uYPNss1TW5NZdW1j6m5d\"></elevenlabs-convai>
    <script src=\"https://elevenlabs.io/convai-widget/index.js\" async type=\"text/javascript\"></script>
    </div>
    """, height=375)

# === Main Chat Interface ===
st.image("logo-wide.png", use_container_width="auto")
st.title("Higher Education Commission Assistant")
st.write("Welcome to the HEC Assistant. How may I assist you with information about higher education policies, programs, or services?")

def get_groq_response(query, context, chat_memory):
    conversation_context = ""
    if chat_memory:
        conversation_context = "Previous conversation:\n" + "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in chat_memory[-5:]
        )

    prompt = f"""You are a professional virtual assistant for the Higher Education Commission (HEC), Pakistan...
    Conversation context: {conversation_context}
    Context: {context}
    Question: {query}
    Answer: """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        stream=True,
    )
    return response

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Your query from HEC Assistant:")

if user_query:
    if st.session_state.current_chat is None:
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = user_query[:30] + "..." if len(user_query) > 30 else user_query
        st.session_state.conversation_memory[chat_id] = []

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    chat_memory = st.session_state.conversation_memory.get(st.session_state.current_chat, [])
    st.session_state.conversation_memory.setdefault(st.session_state.current_chat, []).append({"role": "user", "content": user_query})

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

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_memory.setdefault(st.session_state.current_chat, []).append({"role": "assistant", "content": full_response})
    st.session_state.conversations[st.session_state.current_chat] = st.session_state.messages
