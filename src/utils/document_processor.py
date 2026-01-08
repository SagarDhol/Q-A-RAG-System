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
        """Split text into chunks with metadata, preserving context and structure.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []
        
        # First, split by double newlines to preserve paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # If paragraph is a heading or title (ends with a colon or is in all caps)
            is_heading = paragraph.endswith(':') or (len(paragraph) < 100 and paragraph.isupper())
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and current_length + len(paragraph) > self.chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'length': len(chunk_text)
                })
                
                # Start a new chunk with overlap (last paragraph or two)
                overlap_paragraphs = min(2, len(current_chunk))
                current_chunk = current_chunk[-overlap_paragraphs:]
                current_length = sum(len(p) + 2 for p in current_chunk)  # +2 for newlines
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 2  # +2 for newlines
            
            # If this is a heading, ensure it's at the start of a chunk
            if is_heading and len(current_chunk) > 1:
                # Move the heading to the next chunk
                heading = current_chunk.pop()
                current_length -= len(heading) + 2
                
                # Finalize current chunk if not empty
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': chunk_id,
                        'length': len(chunk_text)
                    })
                
                # Start new chunk with the heading
                current_chunk = [heading]
                current_length = len(heading)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'length': len(chunk_text)
            })
        
        # Post-process to ensure no chunk is too large
        final_chunks = []
        for chunk in chunks:
            if chunk['length'] <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # If a chunk is still too large, split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if current_chunk and current_length + len(sentence) > self.chunk_size:
                        chunk_text = ' '.join(current_chunk)
                        chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                        final_chunks.append({
                            'text': chunk_text,
                            'chunk_id': chunk_id,
                            'length': len(chunk_text)
                        })
                        current_chunk = []
                        current_length = 0
                        
                    current_chunk.append(sentence)
                    current_length += len(sentence) + 1
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                    final_chunks.append({
                        'text': chunk_text,
                        'chunk_id': chunk_id,
                        'length': len(chunk_text)
                    })
        
        return final_chunks
    
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
