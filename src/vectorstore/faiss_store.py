import os
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

class FAISSStore:
    def __init__(self, vector_dim: int, index_path: str):
        """Initialize the FAISS vector store."""
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.index = None
        self.metadata = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            # Create a new FAISS index
            self.index = faiss.IndexFlatL2(self.vector_dim)
    
    def _load_index(self):
        """Load index and metadata from disk."""
        try:
            # Load the FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            metadata_path = f"{self.index_path}.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            print(f"Loaded index with {len(self.metadata)} vectors")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            # If loading fails, create a new index
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.metadata = []
    
    def save(self):
        """Save the index and metadata to disk."""
        if self.index is None:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        metadata_path = f"{self.index_path}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add_embeddings(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add embeddings to the index with associated metadata."""
        if not texts or not embeddings:
            return
            
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
            
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        if self.index.ntotal == 0:
            # First batch - use the embeddings directly
            self.index.add(embeddings_np)
        else:
            # Subsequent batches - check dimension compatibility
            if embeddings_np.shape[1] != self.vector_dim:
                raise ValueError(
                    f"Dimensionality mismatch: "
                    f"expected {self.vector_dim}, got {embeddings_np.shape[1]}"
                )
            self.index.add(embeddings_np)
        
        # Add metadata
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            self.metadata.append({
                'text': text,
                'index': self.index.ntotal - len(texts) + i,
                **metadata
            })
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find the k most similar documents to the query embedding."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_np, k)
        
        # Get the metadata for the top results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            result = self.metadata[idx].copy()
            result['score'] = float(distance)
            results.append(result)
        
        return results
    
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        if doc_id < 0 or doc_id >= len(self.metadata):
            return None
        return self.metadata[doc_id]
    
    def get_documents(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """Get multiple documents by their IDs."""
        return [self.get_document(doc_id) for doc_id in doc_ids]
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the index."""
        return self.metadata.copy()
    
    def clear(self) -> None:
        """Clear the index and all metadata."""
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []
