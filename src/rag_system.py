import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from config import settings
from src.embedding.nomic_embedder import NomicEmbedder
from src.llm.llama3_client import Llama3Client
from src.vectorstore.faiss_store import FAISSStore
from src.utils.document_processor import DocumentProcessor

class RAGSystem:
    """Main RAG system for question answering."""
    
    def __init__(self):
        """Initialize the RAG system with all necessary components."""
        print("Initializing RAG system...")
        
        # Initialize components
        self.embedder = NomicEmbedder(settings.EMBEDDING_MODEL)
        self.llm = Llama3Client(settings.LLM_MODEL)
        self.vector_store = FAISSStore(
            vector_dim=self.embedder.embedding_dimension,
            index_path=settings.VECTOR_STORE_PATH
        )
        self.document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        print(f"RAG system initialized with model: {settings.LLM_MODEL}")
    
    def ingest_documents(self, directory: Optional[Path] = None) -> Dict[str, Any]:
        """Ingest documents from a directory into the vector store.
        
        Args:
            directory: Directory containing documents to ingest. Uses settings.DOCUMENTS_DIR if None.
            
        Returns:
            Dictionary with ingestion statistics
        """
        if directory is None:
            directory = settings.DOCUMENTS_DIR
        
        print(f"Ingesting documents from: {directory}")
        
        # Process all documents in the directory
        chunks = self.document_processor.process_directory(directory)
        
        if not chunks:
            return {"status": "error", "message": "No documents found to process"}
        
        print(f"Processed {len(chunks)} chunks from documents")
        
        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {k: v for k, v in chunk.items() if k != "text"}
            for chunk in chunks
        ]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.embed_documents(texts)
        
        # Add to vector store
        print("Adding to vector store...")
        self.vector_store.add_embeddings(texts, embeddings, metadatas)
        self.vector_store.save()
        
        return {
            "status": "success",
            "chunks_processed": len(chunks),
            "total_vectors": self.vector_store.index.ntotal
        }
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: The question to ask
            top_k: Number of relevant chunks to retrieve (defaults to settings.TOP_K_RESULTS)
            
        Returns:
            Dictionary containing the answer and relevant context
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(question)
        
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.similarity_search(query_embedding, k=top_k)
        
        # Format context
        context = "\n\n".join(
            f"[Document {i+1}, Score: {chunk['score']:.2f}]\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        )
        
        # Generate prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question based on the context above. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
        
        Provide a concise and accurate answer:"""
        
        # Generate response
        answer = self.llm.generate(prompt)
        
        # Prepare sources
        sources = [
            {
                "document": chunk.get("document_name", "Unknown"),
                "score": chunk.get("score", 0.0),
                "text": chunk.get("text", "")[:200] + "..."  # Truncate long texts
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def query_structured(
        self, 
        question: str, 
        response_format: Dict[str, Any],
        top_k: int = None
    ) -> Dict[str, Any]:
        """Query the RAG system with a question and get a structured response.
        
        Args:
            question: The question to ask
            response_format: JSON schema for the desired response format
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing the structured answer and relevant context
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(question)
        
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.similarity_search(query_embedding, k=top_k)
        
        # Format context
        context = "\n\n".join(
            f"[Document {i+1}, Score: {chunk['score']:.2f}]\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        )
        
        # Generate prompt for structured response
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question based on the context above. If the context doesn't contain the answer, 
        indicate this in your response.
        """
        
        # Generate structured response
        response = self.llm.generate_structured(prompt, response_format)
        
        # Prepare sources
        sources = [
            {
                "document": chunk.get("document_name", "Unknown"),
                "score": chunk.get("score", 0.0),
                "text": chunk.get("text", "")[:200] + "..."  # Truncate long texts
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "question": question,
            "answer": response,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear the vector store index.
        
        Returns:
            Status of the operation
        """
        self.vector_store.clear()
        self.vector_store.save()
        return {"status": "success", "message": "Vector store index cleared"}
