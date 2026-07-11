"""
Vector store for resume embeddings using sentence-transformers and ChromaDB
"""
import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeVectorStore:
    """
    Handles embedding generation and vector storage for resumes
    """
    
    def __init__(self, collection_name: str = "resumes", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store with embedding model and ChromaDB client.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize the embedding model
        logger.info("Loading sentence-transformers model...")
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        logger.info("Model loaded successfully!")
        
        # Initialize ChromaDB client - using older API
        try:
            # Try the newer API first
            self.client = chromadb.PersistentClient(path=persist_directory)
        except:
            # Fallback to older API
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
        
        # Get or create collection - using older API
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=collection_name)
        except:
            # Create new collection
            self.collection = self.client.create_collection(name=collection_name)
        
        logger.info(f"Connected to ChromaDB collection: {collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def add_resume(self, resume_text: str, metadata: Dict[str, Any]) -> str:
        """Add a single resume to the vector store."""
        doc_id = str(uuid.uuid4())
        embedding = self.generate_embeddings([resume_text])[0]
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[resume_text],
            metadatas=[metadata]
        )
        
        logger.info(f"Added resume with ID: {doc_id}")
        return doc_id
    
    def add_resumes_batch(self, resumes: List[Dict[str, Any]]) -> List[str]:
        """Add multiple resumes to the vector store in batch."""
        ids = [str(uuid.uuid4()) for _ in range(len(resumes))]
        texts = [r['text'] for r in resumes]
        metadatas = [r['metadata'] for r in resumes]
        
        embeddings = self.generate_embeddings(texts)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(ids)} resumes to collection")
        return ids
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for resumes similar to the query text."""
        query_embedding = self.generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def get_all_resumes(self) -> Dict[str, Any]:
        """Get all resumes stored in the collection."""
        return self.collection.get()
    
    def delete_resume(self, doc_id: str) -> None:
        """Delete a resume by its ID."""
        self.collection.delete(ids=[doc_id])
        logger.info(f"Deleted resume with ID: {doc_id}")
    
    def count(self) -> int:
        """Get the total number of resumes in the store."""
        return self.collection.count()
    
    def clear_all(self) -> None:
        """Delete all resumes from the store."""
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)
            logger.info(f"Cleared {len(all_ids)} resumes from collection")