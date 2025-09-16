from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import numpy as np

from src.chunking.base import Chunk


class EmbeddingResult:
    def __init__(
        self,
        chunk_id: str,
        embedding: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        if isinstance(embedding, list):
            self.embedding = np.array(embedding, dtype=float)
        else:
            self.embedding = embedding
            
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"EmbeddingResult(chunk_id={self.chunk_id}, dim={len(self.embedding)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata
        }


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_chunk(self, chunk: Chunk) -> EmbeddingResult:
        pass
    
    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass