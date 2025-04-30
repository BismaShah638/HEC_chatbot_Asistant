import streamlit as st
import os
import time
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from streamlit.components.v1 import html

# Initialize Groq client
client = Groq(api_key="your-groq-api-key")  # Replace with env variable for security

# Streamlit config
st.set_page_config(page_title="HEC Assistant", page_icon="logo.png", layout="centered", initial_sidebar_state="collapsed")

# Session state init
for key in ["messages", "conversations", "current_chat", "chat_titles", "conversation_memory"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "conversation" in key else []

# Load documents
def load_documents():
    documents = []
    for file in os.listdir("./Data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"./Data/{file}")
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(f"./Data/{file}")
            documents.extend(loader.load())
    return documents

# Split into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Embeddings + FAISS
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db_path = "faiss_index"
if not os.path.exists(db_path):
    docs = load_documents()
    chunks = split_documents(docs)
    db = FAISS.from_documents(chunks, embedding=embeddings)
    db.save_local(db_path)
else:
    db = FAISS.load_local(db_path, embeddings)

# Sidebar UI
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

# Header
st.image("logo-wide.png", use_container_width="auto")
st.title("Higher Education Commission Assistant")
st.write("Welcome to the HEC Assistant. How may I assist you with information about higher education policies, programs, or services?")

# Generate response from Groq
def get_groq_response(query, context, chat_memory):
    context_str = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in chat_memory[-5:]]) if chat_memory else ""
    prompt = f"""
You are a professional virtual assistant for HEC Pakistan. Use the context and chat history to give accurate, clear, and concise answers.

Conversation history:
{context_str}

Context:
{context}

Question: {query}

Answer:
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        stream=True,
    )
    return response

# Chat display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input
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
    st.session_state.conversation_memory[st.session_state.current_chat].append({"role": "user", "content": user_query})
    results = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in get_groq_response(user_query, context, chat_memory):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_memory[st.session_state.current_chat].append({"role": "assistant", "content": full_response})
    st.session_state.conversations[st.session_state.current_chat] = st.session_state.messages
