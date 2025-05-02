import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from streamlit.components.v1 import html

# Streamlit page setup
st.set_page_config(page_title="HEC Assistant", page_icon="📘", layout="centered", initial_sidebar_state="collapsed")

# Session state setup
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

# Load PDF and DOCX documents
def load_documents():
    documents = []
    for file in os.listdir():
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file)
            documents.extend(loader.load())
    return documents

# Text splitter
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Vector DB setup
VECTOR_DB_PATH = "chroma_db"
embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)

if not os.path.exists(VECTOR_DB_PATH):
    with st.spinner("Indexing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DB_PATH)
        db.persist()
else:
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

# Sidebar (Chat history)
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

    html("""
    <div style="margin-top: 30px;">
    <elevenlabs-convai agent-id="reJNIPHhfFDU9lBMg1z0"></elevenlabs-convai>
    <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
    </div>
    """, height=375)

# Header
st.title("🎓 HEC Assistant")
st.write("Ask about policies, degrees, scholarships, or any other information related to HEC Pakistan.")

# LLM initialization
llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model="llama3-70b-8192")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Your query:")

def get_response(query, context, memory):
    history = "\n".join([f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in memory[-5:]])
    prompt = f"""You are a helpful assistant for HEC Pakistan. Respond clearly and formally based only on the given context.

Context:\n{context}
{history}
User: {query}
Assistant:"""
    return llm.invoke(prompt)

if user_input:
    if st.session_state.current_chat is None:
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = user_input[:30] + "..." if len(user_input) > 30 else user_input
        st.session_state.conversation_memory[chat_id] = []

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    memory = st.session_state.conversation_memory.get(st.session_state.current_chat, [])
    st.session_state.conversation_memory[st.session_state.current_chat].append({"role": "user", "content": user_input})

    docs = db.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = get_response(user_input, context, memory)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.conversation_memory[st.session_state.current_chat].append({"role": "assistant", "content": response})
    st.session_state.conversations[st.session_state.current_chat] = st.session_state.messages
