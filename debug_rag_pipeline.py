import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import pandas as pd
import time
from datetime import datetime

import streamlit as st
import os

# Access your secret API key using the Streamlit secret manager
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)



# Initialize embeddings and Chroma DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./chroma_db"
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def rewrite_query(query):
    """Debug the query rewriting step."""
    prompt = f"""You are an AI assistant helping to rewrite user queries to improve retrieval in a RAG system.
    
    Your task is to rewrite the given query to make it more specific, clear, and optimized for semantic search. Use the following high-value keywords to improve the relevance and specificity of the query **only as synonyms or clarifying substitutions for terms already present in the original query**.
    
    Do **not fabricate, assume, or add any new information or intent not present in the original query**. Do **not deviate** from the original query’s meaning or purpose. Expand acronyms (e.g., HEC = Higher Education Commission) only if they are used in the original query. The rewritten version must stay faithful to the original query while improving clarity and keyword alignment.
    
    High-value keywords: HEC, attestation, documents, education, policy, degree, equivalence, applicants, case, courier, plagiarism, application, students, transcript, urgent, certificates.
    
    Only return the rewritten query without any explanations or additional text.
    
    Original query: {query}
    
    Rewritten query: """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=200,
        stream=False,
    )
    
    return response.choices[0].message.content.strip()

def get_relevant_chunks(query, k=3):
    """Debug the document retrieval step."""
    return db.similarity_search_with_relevance_scores(query, k=k)

def get_groq_response(query, context, chat_memory=None):
    """Debug the LLM response generation step."""
    conversation_context = ""
    if chat_memory and len(chat_memory) > 0:
        conversation_context = "Previous conversation:\n"
        for msg in chat_memory[-5:]:
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
        model="llama3-70b-8192",
        temperature=0.3,
        stream=False,
    )
    
    return response.choices[0].message.content.strip()

def display_chunk_info(doc, score):
    """Display detailed information about a document chunk."""
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Content")
        st.text(doc.page_content)
    with col2:
        st.markdown("### Metadata")
        st.write(f"**Score:** {score:.4f}")
        for key, value in doc.metadata.items():
            st.write(f"**{key}:** {value}")

def save_debug_log(debug_data):
    """Save debug information to a log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "debug_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    filename = f"{log_dir}/rag_debug_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== RAG Pipeline Debug Log ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Original Query: {debug_data['original_query']}\n")
        f.write(f"Rewritten Query: {debug_data['rewritten_query']}\n\n")
        f.write("=== Retrieved Chunks ===\n")
        for i, (doc, score) in enumerate(debug_data['chunks'], 1):
            f.write(f"\nChunk {i} (Score: {score:.4f})\n")
            f.write("Content:\n")
            f.write(doc.page_content + "\n")
            f.write("Metadata:\n")
            for key, value in doc.metadata.items():
                f.write(f"{key}: {value}\n")
        f.write("\n=== Final Response ===\n")
        f.write(debug_data['final_response'])
    
    return filename

def main():
    st.title("RAG Pipeline Debugger")
    st.markdown("Inspect each component of the RAG system.")
    
    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Debug controls
    with st.sidebar:
        st.title("Debug Controls")
        k_value = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
        show_history = st.checkbox("Include chat history in context", value=True)
        save_logs = st.checkbox("Save debug logs to file", value=True)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Query input
    query = st.text_input("Enter your query:")
    
    if query:
        debug_data = {
            'original_query': query,
            'chunks': None,
            'rewritten_query': None,
            'final_response': None
        }
        
        # Step 1: Query Rewriting
        st.markdown("## 1. Query Rewriting")
        with st.spinner("Rewriting query..."):
            rewritten = rewrite_query(query)
            debug_data['rewritten_query'] = rewritten
            st.write("**Original Query:**", query)
            st.write("**Rewritten Query:**", rewritten)
        
        # Step 2: Document Retrieval
        st.markdown("## 2. Document Retrieval")
        with st.spinner("Retrieving relevant chunks..."):
            results = get_relevant_chunks(rewritten, k=k_value)
            debug_data['chunks'] = results
            
            # Display results in a table format
            data = []
            for i, (doc, score) in enumerate(results, 1):
                data.append({
                    "Chunk #": i,
                    "Score": f"{score:.4f}",
                    "Source": doc.metadata.get("source", "N/A"),
                    "Page": doc.metadata.get("page", "N/A"),
                    "Preview": doc.page_content[:100] + "..."
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df)
            
            # Detailed chunk analysis in expandable sections
            for i, (doc, score) in enumerate(results, 1):
                with st.expander(f"Chunk {i} Details"):
                    display_chunk_info(doc, score)
        
        # Step 3: Response Generation
        st.markdown("## 3. Response Generation")
        with st.spinner("Generating response..."):
            # Prepare context from chunks
            context = "\n".join([doc.page_content for doc, _ in results])
            
            # Get chat history if enabled
            chat_memory = st.session_state.chat_history if show_history else None
            
            # Generate response
            response = get_groq_response(query, context, chat_memory)
            debug_data['final_response'] = response
            
            st.markdown("### Final Response:")
            st.write(response)
        
        # Save debug logs if enabled
        if save_logs:
            log_file = save_debug_log(debug_data)
            st.sidebar.success(f"Debug log saved to: {log_file}")
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        if show_history:
            st.markdown("## Chat History")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

if __name__ == "__main__":
    main() 
