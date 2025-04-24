import numpy as np
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt

# Initialize embeddings and Chroma DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./chroma_db"
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def get_all_documents():
    """Retrieve all documents from the Chroma database."""
    # Using a generic query to retrieve documents
    results = db.similarity_search_with_relevance_scores("", k=1000)
    return [doc for doc, _ in results]

def extract_keywords(documents: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    """Extract keywords using TF-IDF."""
    # Combine all documents into a single text corpus
    corpus = [doc.page_content for doc in documents]
    
    # Initialize TF-IDF
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=100,
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ characters
    )
    
    # Fit and transform the corpus
    tfidf_matrix = tfidf.fit_transform(corpus)
    
    # Calculate average TF-IDF scores for each term
    avg_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    # Get feature names and their scores
    feature_names = np.array(tfidf.get_feature_names_out())
    keyword_scores = list(zip(feature_names, avg_tfidf))
    
    # Sort by score and return top N
    return sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_n]

def analyze_metadata(documents: List[str]) -> dict:
    """Analyze metadata from documents."""
    sources = Counter()
    total_docs = len(documents)
    
    for doc in documents:
        if 'source' in doc.metadata:
            sources[doc.metadata['source']] += 1
    
    return {
        'total_documents': total_docs,
        'unique_sources': len(sources),
        'source_distribution': dict(sources.most_common())
    }

def main():
    print("Loading documents from Chroma DB...")
    documents = get_all_documents()
    
    if not documents:
        print("No documents found in the database!")
        return
    
    print(f"\nAnalyzing {len(documents)} documents...")
    
    # Extract and display keywords
    keywords = extract_keywords(documents)
    print("\nTop 20 Keywords by TF-IDF Score:")
    print("-" * 40)
    for keyword, score in keywords:
        print(f"{keyword:<20} {score:.4f}")
    
    # Analyze metadata
    metadata_stats = analyze_metadata(documents)
    print("\nDatabase Statistics:")
    print("-" * 40)
    print(f"Total Documents: {metadata_stats['total_documents']}")
    print(f"Unique Sources: {metadata_stats['unique_sources']}")
    
    print("\nSource Distribution:")
    print("-" * 40)
    for source, count in metadata_stats['source_distribution'].items():
        print(f"{source}: {count} documents")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    keywords_df = pd.DataFrame(keywords, columns=['Term', 'Score'])
    plt.bar(keywords_df['Term'][:10], keywords_df['Score'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Keywords by TF-IDF Score')
    plt.tight_layout()
    plt.savefig('keyword_analysis.png')
    print("\nVisualization saved as 'keyword_analysis.png'")

if __name__ == "__main__":
    main() 
