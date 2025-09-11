from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.document_processing.base import Document

class Chunk:
    def __init__(
            self,
            text: str,
            doc_id: str,
            chunk_id: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.metadata = metadata if metadata is not None else {}
    
    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, doc_id={self.doc_id}, text_len={self.text.__len__()})"

class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(
            self,
            document: Document,
            chunk_size: int,
            overlap: int = 0
    ) -> List[Chunk]:
        """Chunk a document into smaller pieces."""
        pass