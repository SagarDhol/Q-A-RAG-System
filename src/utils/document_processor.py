from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import hashlib
from tqdm import tqdm

class DocumentProcessor:
    """Process documents for the RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, file_path: Path) -> str:
        """Load text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            The text content of the document
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []
        
        # Simple sentence splitting on common sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the chunk size, finalize the current chunk
            if current_chunk and current_length + sentence_length > self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'length': len(chunk_text)
                })
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap // 20)  # Approximate word count
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) + 1 for s in current_chunk)  # +1 for spaces
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'length': len(chunk_text)
            })
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a document file into chunks with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            # Load the document
            text = self.load_document(file_path)
            
            # Split into chunks
            chunks = self.split_into_chunks(text)
            
            # Add document metadata
            for chunk in chunks:
                chunk.update({
                    'document_id': file_path.stem,
                    'document_path': str(file_path),
                    'document_name': file_path.name,
                })
            
            return chunks
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory: Path, file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process all documents in a directory.
        
        Args:
            directory: Directory containing documents
            file_extensions: List of file extensions to include (e.g., ['.txt', '.md'])
            
        Returns:
            List of all document chunks with metadata
        """
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf']
            
        all_chunks = []
        
        # Get all files with specified extensions
        files = []
        for ext in file_extensions:
            files.extend(directory.glob(f'*{ext}'))
        
        # Process each file
        for file_path in tqdm(files, desc="Processing documents"):
            chunks = self.process_document(file_path)
            all_chunks.extend(chunks)
        
        return all_chunks
