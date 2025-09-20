import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
from collections import Counter

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.vector_store.base import BaseVectorStore, VectorStoreResult
from src.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining semantic and keyword-based search."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance for query encoding
            semantic_weight: Weight for semantic similarity scores
            keyword_weight: Weight for keyword matching scores
            similarity_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.similarity_threshold = similarity_threshold
        
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            self.semantic_weight = semantic_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (remove stopwords, punctuation)
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'how'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, query_keywords: List[str], chunk_text: str) -> float:
        """Calculate keyword matching score using TF-IDF-like approach."""
        if not query_keywords or not chunk_text:
            return 0.0
        
        chunk_keywords = self._extract_keywords(chunk_text)
        if not chunk_keywords:
            return 0.0
        
        # Calculate keyword frequencies
        chunk_freq = Counter(chunk_keywords)
        query_freq = Counter(query_keywords)
        
        # Calculate score based on keyword overlap
        score = 0.0
        for keyword in query_keywords:
            if keyword in chunk_freq:
                # TF-IDF-like score: term frequency * inverse document frequency
                tf = chunk_freq[keyword] / len(chunk_keywords)
                # Simplified IDF (could be improved with corpus statistics)
                idf = 1.0 + (1.0 / (1.0 + chunk_freq[keyword]))
                score += tf * idf * query_freq[keyword]
        
        # Normalize by query length
        return score / len(query_keywords)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve documents using hybrid approach."""
        start_time = datetime.now()
        
        try:
            # Step 1: Semantic retrieval (get more candidates for reranking)
            initial_k = min(top_k * 3, 50)  # Get more candidates
            query_embedding = self.embedder.embed_query(query)
            
            semantic_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=initial_k,
                filters=filters
            )
            
            # Step 2: Extract query keywords
            query_keywords = self._extract_keywords(query)
            
            # Step 3: Calculate hybrid scores
            hybrid_results = []
            for result in semantic_results:
                if not result.chunk or not result.chunk.text:
                    continue
                
                # Get semantic score
                semantic_score = result.score
                
                # Calculate keyword score
                keyword_score = self._calculate_keyword_score(
                    query_keywords, 
                    result.chunk.text
                )
                
                # Combine scores
                hybrid_score = (
                    self.semantic_weight * semantic_score +
                    self.keyword_weight * keyword_score
                )
                
                # Create new result with hybrid score
                hybrid_result = VectorStoreResult(
                    chunk_id=result.chunk_id,
                    score=hybrid_score,
                    chunk=result.chunk,
                    metadata={
                        **result.metadata,
                        'semantic_score': semantic_score,
                        'keyword_score': keyword_score,
                        'hybrid_score': hybrid_score
                    }
                )
                hybrid_results.append(hybrid_result)
            
            # Step 4: Sort by hybrid score and take top_k
            hybrid_results.sort(key=lambda x: x.score, reverse=True)
            final_results = hybrid_results[:top_k]
            
            # Step 5: Filter by similarity threshold
            filtered_results = [
                result for result in final_results
                if result.score >= self.similarity_threshold
            ]
            
            # Calculate retrieval time
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                "strategy": "hybrid",
                "retrieval_time": retrieval_time,
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
                "query_keywords": query_keywords,
                "initial_candidates": len(semantic_results),
                "final_results": len(filtered_results),
                "similarity_threshold": self.similarity_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(
                f"Hybrid retrieval: {len(filtered_results)} results "
                f"in {retrieval_time:.3f}s for query: {query[:100]}"
            )
            
            return RetrievalResult(
                query=query,
                results=filtered_results,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise
