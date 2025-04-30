import streamlit as st
import os
import time
import zipfile
import requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit.components.v1 import html

# === 1. Extract Data.zip from GitHub ===
DATA_DIR = "./Data"
ZIP_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/raw/main/data.zip"  # ⬅️ CHANGE THIS

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = "data.zip"
    with open(zip_path, "wb") as f:
        f.write(requests.get(ZIP_URL).content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)

# === 2. Setup Streamlit Page ===
st.set_page_config(page_title="HEC Assistant", page_icon="📘", layout="centered", initial_sidebar_state="collapsed")

# === 3. Session State Init ===
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

# === 4. Load and Process Documents ===
def load_documents():
    documents = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif file.endswith(".docx"):
            documents.extend(Docx2txtLoader(path).load())
    return documents

def split_documents(documents):
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

# === 5. Embeddings & Chroma DB Init ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    documents = load_documents()
    chunks = split_documents(documents)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# === 6. Sidebar Chat History ===
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
        title = st.session_state.chat_titles.get(chat_id, "New Chat")
        if st.button(title, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat = chat_id
            st.session_state.messages = st.session_state.conversations[chat_id]
            st.rerun()
    st.divider()

    # ElevenLabs Widget Embed (Optional)
    html("""
    <div style="margin-top: 30px;">
    <elevenlabs-convai agent-id="uYPNss1TW5NZdW1j6m5d"></elevenlabs-convai>
    <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
    </div>
    """, height=375)

# === 7. Main Chat UI ===
st.title("🎓 Higher Education Commission Assistant")
st.write("Welcome to the HEC Assistant. How may I assist you with higher education policies, programs, or services?")

# === 8. Groq Response Function ===
client = Groq(api_key="YOUR_GROQ_API_KEY")  # ⬅️ Replace with secure env var if needed

def get_groq_response(query, context, chat_memory):
    conversation_context = "\n".join([f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_memory[-5:]])
    
    prompt = f"""
    You are a professional virtual assistant for the Higher Education Commission (HEC), Pakistan...
    Conversation context: {conversation_context}
    Context: {context}
    Question: {query}
    Answer:
    """
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        stream=True,
    )

# === 9. Chat Interaction ===
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

    st.session_state.conversation_memory[st.session_state.current_chat].append({"role": "user", "content": user_query})

    results = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in get_groq_response(user_query, context, st.session_state.conversation_memory[st.session_state.current_chat]):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_memory[st.session_state.current_chat].append({"role": "assistant", "content": full_response})
    st.session_state.conversations[st.session_state.current_chat] = st.session_state.messages
