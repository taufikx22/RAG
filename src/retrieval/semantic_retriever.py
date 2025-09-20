import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.vector_store.base import BaseVectorStore
from src.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    """Semantic retrieval using vector similarity search."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance for query encoding
            similarity_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve documents using semantic similarity."""
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Search vector store
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.score >= self.similarity_threshold
            ]
            
            # Calculate retrieval time
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                "strategy": "semantic",
                "retrieval_time": retrieval_time,
                "total_candidates": len(search_results),
                "filtered_results": len(filtered_results),
                "similarity_threshold": self.similarity_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(
                f"Semantic retrieval: {len(filtered_results)} results "
                f"in {retrieval_time:.3f}s for query: {query[:100]}"
            )
            
            return RetrievalResult(
                query=query,
                results=filtered_results,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            raise
