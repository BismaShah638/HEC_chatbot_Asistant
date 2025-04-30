from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize app
app = Flask(__name__)
CORS(app)

# Load Chroma DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./chroma_db"
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# RAG + LLM
def get_groq_response(query, context):
    prompt = f"""You are an HEC assistant. Use only the context to answer clearly and formally.

Context: {context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        stream=False,
    )
    return response.choices[0].message.content.strip()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    answer = get_groq_response(query, context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
