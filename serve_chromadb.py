from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from flask_cors import CORS
# Initialize Flask App
app = Flask(__name__)
CORS(app)
# Load ChromaDB from the persisted directory
PERSIST_DIR = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text") 
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

@app.route('/api/query', methods=['POST'])
def query_chroma_db():
    data = request.json
    query_text = data.get('query')
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    # Perform similarity search
    results = db.similarity_search(query_text, top_k=3)
    documents = [doc.page_content for doc in results]
    return jsonify({"documents": documents})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
