from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from src.vector_store.base import VectorStoreResult


class RetrievalResult:
    """Class representing a retrieval result."""
    
    def __init__(
        self,
        query: str,
        results: List[VectorStoreResult],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a retrieval result.
        
        Args:
            query: The original query
            results: List of retrieved results
            metadata: Additional metadata about the retrieval
        """
        self.query = query
        self.results = results
        self.metadata = metadata or {}
    
    def get_context(self) -> str:
        """Get concatenated context from all results."""
        contexts = []
        for result in self.results:
            if result.chunk and result.chunk.text:
                contexts.append(result.chunk.text)
        return "\n\n".join(contexts)
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __repr__(self) -> str:
        return f"RetrievalResult(query='{self.query[:50]}...', results={len(self.results)})"


class BaseRetriever(ABC):
    """Base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Retrieval result object
        """
        pass
