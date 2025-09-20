import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
from sentence_transformers import CrossEncoder

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.vector_store.base import BaseVectorStore, VectorStoreResult
from src.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class RerankingRetriever(BaseRetriever):
    """Retrieval with cross-encoder reranking for improved relevance."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k: int = 20,
        similarity_threshold: float = 0.0,
        device: Optional[str] = None
    ):
        """
        Initialize reranking retriever.
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance for query encoding
            rerank_model: Cross-encoder model for reranking
            initial_k: Number of candidates to retrieve before reranking
            similarity_threshold: Minimum similarity score threshold
            device: Device to run reranking model on
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.initial_k = initial_k
        self.similarity_threshold = similarity_threshold
        
        try:
            # Initialize cross-encoder for reranking
            self.cross_encoder = CrossEncoder(rerank_model, device=device)
            logger.info(f"Initialized cross-encoder: {rerank_model}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {str(e)}")
            self.cross_encoder = None
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve documents with reranking."""
        start_time = datetime.now()
        
        try:
            # Step 1: Initial semantic retrieval
            query_embedding = self.embedder.embed_query(query)
            
            initial_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.initial_k,
                filters=filters
            )
            
            if not initial_results:
                logger.warning("No initial results found")
                return RetrievalResult(
                    query=query,
                    results=[],
                    metadata={"strategy": "reranking", "error": "no_initial_results"}
                )
            
            # Step 2: Reranking with cross-encoder (if available)
            if self.cross_encoder:
                reranked_results = self._rerank_with_cross_encoder(
                    query, initial_results
                )
            else:
                logger.warning("Cross-encoder not available, using semantic scores")
                reranked_results = initial_results
            
            # Step 3: Take top_k results
            final_results = reranked_results[:top_k]
            
            # Step 4: Filter by similarity threshold
            filtered_results = [
                result for result in final_results
                if result.score >= self.similarity_threshold
            ]
            
            # Calculate retrieval time
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                "strategy": "reranking",
                "retrieval_time": retrieval_time,
                "rerank_model": getattr(self.cross_encoder, 'model_name', 'none'),
                "initial_candidates": len(initial_results),
                "final_results": len(filtered_results),
                "similarity_threshold": self.similarity_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(
                f"Reranking retrieval: {len(filtered_results)} results "
                f"in {retrieval_time:.3f}s for query: {query[:100]}"
            )
            
            return RetrievalResult(
                query=query,
                results=filtered_results,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in reranking retrieval: {str(e)}")
            raise
    
    def _rerank_with_cross_encoder(
        self, 
        query: str, 
        results: List[VectorStoreResult]
    ) -> List[VectorStoreResult]:
        """Rerank results using cross-encoder."""
        if not results or not self.cross_encoder:
            return results
        
        try:
            # Prepare query-document pairs for reranking
            pairs = []
            valid_results = []
            
            for result in results:
                if result.chunk and result.chunk.text:
                    pairs.append([query, result.chunk.text])
                    valid_results.append(result)
            
            if not pairs:
                logger.warning("No valid pairs for reranking")
                return results
            
            # Get reranking scores
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Create new results with reranking scores
            reranked_results = []
            for i, (result, score) in enumerate(zip(valid_results, rerank_scores)):
                # Convert score to float if it's a tensor
                if torch.is_tensor(score):
                    score = score.item()
                
                new_result = VectorStoreResult(
                    chunk_id=result.chunk_id,
                    score=float(score),
                    chunk=result.chunk,
                    metadata={
                        **result.metadata,
                        'original_score': result.score,
                        'rerank_score': float(score)
                    }
                )
                reranked_results.append(new_result)
            
            # Sort by reranking score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"Reranked {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results  # Fall back to original results
