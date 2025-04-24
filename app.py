import streamlit as st
import os
import time
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit.components.v1 import html
import streamlit as st
import os

# Access your secret API key using the Streamlit secret manager
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)


# Initialize Groq client
# client = Groq(
   #  api_key="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"
# )

# Set up Streamlit page
st.set_page_config(page_title = "HEC Assistant", page_icon = "logo.png", layout="centered", initial_sidebar_state = "collapsed")

# Initialize session state for chat history and conversations
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

# Load and process documents
def load_documents():
    documents = []
    data_path = "./Data"

    if not os.path.exists(data_path):
        st.warning("⚠️ 'Data' folder not found. Skipping document loading.")
        return documents

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join(data_path, file))
            documents.extend(loader.load())

    return documents


# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create or load Chroma database
persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    documents = load_documents()
    chunks = split_documents(documents)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    db.persist()
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Create sidebar for chat history
with st.sidebar:
    st.title("Chat History")
    
    # New Chat button
    if st.button("+ New Chat", use_container_width=True):
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.messages = []
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = "New Chat"
        st.session_state.conversation_memory[chat_id] = []
        st.rerun()

    # Display divider
    st.divider()

    # Display chat history
    for chat_id in reversed(list(st.session_state.conversations.keys())):
        chat_title = st.session_state.chat_titles.get(chat_id, "New Chat")
        
        # Create a button for each chat
        if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat = chat_id
            st.session_state.messages = st.session_state.conversations[chat_id]
            st.rerun()

    # Display divider
    st.divider()
    # Embed the ElevenLabs voice agent widget
    html_code = """
    <div style="margin-top: 30px;">
    <elevenlabs-convai agent-id=\"uYPNss1TW5NZdW1j6m5d\"></elevenlabs-convai>
    <script src=\"https://elevenlabs.io/convai-widget/index.js\" async type=\"text/javascript\"></script>
    </div>
    """
    html(html_code, height=375)


# Main chat area
st.image("logo-wide.png", use_container_width="auto")
st.title("Higher Education Commission Assistant")
st.write("Welcome to the HEC Assistant. How may I assist you with information about higher education policies, programs, or services?")

# Function to get response from Groq
def get_groq_response(query, context, chat_memory):
    # Prepare conversation history for context
    conversation_context = ""
    if chat_memory and len(chat_memory) > 0:
        conversation_context = "Previous conversation:\n"
        for msg in chat_memory[-5:]:  # Use last 5 exchanges for context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
    
    prompt = f"""You are a professional virtual assistant for the Higher Education Commission (HEC), Pakistan. Your primary role is to use the provided context and respond with accurate, concise, and helpful information regarding higher education programs, services, and policies offered by HEC. You should respond to user inquiries in a friendly yet formal manner, ensuring clarity and professionalism. Avoid mentioning yourself or your role in the responses. 
    
    Only use the information provided in the context or conversation history to answer the question. **Do not fabricate or assume any details, and do not generate URLs or external references unless they are explicitly included in the context.** If the answer cannot be derived from the given information, politely state that there is not enough information to provide an accurate answer and suggest contacting HEC directly for further assistance.
    
    Conversation context: {conversation_context}
    
    Context: {context}
    
    Question: {query}
    
    Answer: """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", #llama3-70b-8192
        temperature=0.3,
        stream=True,
    )
    
    return response

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_query = st.chat_input("Your query from HEC Assistant:")

if user_query:
    # If no current chat, create one
    if st.session_state.current_chat is None:
        chat_id = str(int(time.time()))
        st.session_state.current_chat = chat_id
        st.session_state.conversations[chat_id] = []
        st.session_state.chat_titles[chat_id] = user_query[:30] + "..." if len(user_query) > 30 else user_query
        st.session_state.conversation_memory[chat_id] = []

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Get chat memory for current conversation
    chat_memory = st.session_state.conversation_memory.get(st.session_state.current_chat, [])
    
    # Add user query to conversation memory
    st.session_state.conversation_memory.setdefault(st.session_state.current_chat, []).append({"role": "user", "content": user_query})
    
    # Get relevant documents from vector store using the original query
    results = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    
    # Get response from Groq
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in get_groq_response(user_query, context, chat_memory):
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
        
        response_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.conversation_memory.setdefault(st.session_state.current_chat, []).append({"role": "assistant", "content": full_response})

    # Update the conversations dictionary
    st.session_state.conversations[st.session_state.current_chat] = st.session_state.messages

