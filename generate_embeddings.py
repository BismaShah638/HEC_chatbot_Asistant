import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load and process documents
def load_documents():
    documents = []
    
    # Load PDF
    for file in os.listdir("./data"):
        if file.endswith(".pdf"):
            print(f"Processing PDF file: {file}")
            loader = PyPDFLoader(f"./Data/{file}")
            documents.extend(loader.load())
            
        # Load DOCX
        elif file.endswith(".docx"):
            print(f"Processing DOCX file: {file}")
            loader = Docx2txtLoader(f"./Data/{file}")
            documents.extend(loader.load())
    
    return documents

# Split documents into chunks
def split_documents(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Initialize embeddings
print("Initializing embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create Chroma database
print("Creating Chroma database...")
persist_directory = "./chroma_db"
documents = load_documents()
chunks = split_documents(documents)
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)
print("Process completed successfully! Database has been created and persisted.")
