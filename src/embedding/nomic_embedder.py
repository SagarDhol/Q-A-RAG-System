from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class NomicEmbedder:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """Initialize the Nomic embedding model."""
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        # nomic-embed-text-v1.5 uses 768 dimensions
        return 768
