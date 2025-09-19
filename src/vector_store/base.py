from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from src.embedding.base import EmbeddingResult
from src.chunking.base import Chunk


class VectorStoreResult:
    def __init__(
        self,
        chunk_id: str,
        score: float,
        chunk: Optional[Chunk] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.score = score
        self.chunk = chunk
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"VectorStoreResult(chunk_id={self.chunk_id}, score={self.score})"


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, embedding: List[EmbeddingResult]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List["VectorStoreResult"]:
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> bool:
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        pass
