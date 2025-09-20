from typing import Dict, Type, Optional
import logging

from src.retrieval.base import BaseRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranking_retriever import RerankingRetriever
from src.vector_store.base import BaseVectorStore
from src.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating retriever instances."""
    
    def __init__(self, config=None):
        """
        Initialize retriever factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._retrievers: Dict[str, Type[BaseRetriever]] = {}
        
        # Register default retrievers
        self.register_retriever("semantic", SemanticRetriever)
        self.register_retriever("hybrid", HybridRetriever)  
        self.register_retriever("reranking", RerankingRetriever)
    
    def register_retriever(self, name: str, retriever_class: Type[BaseRetriever]) -> None:
        """Register a retriever class."""
        self._retrievers[name.lower()] = retriever_class
        logger.debug(f"Registered retriever {retriever_class.__name__} as '{name}'")
    
    def get_retriever(
        self,
        strategy: str,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder
    ) -> Optional[BaseRetriever]:
        """
        Get a retriever instance.
        
        Args:
            strategy: Retrieval strategy name
            vector_store: Vector store instance
            embedder: Embedder instance
            
        Returns:
            Configured retriever instance or None
        """
        strategy = strategy.lower()
        retriever_class = self._retrievers.get(strategy)
        
        if not retriever_class:
            logger.error(f"No retriever found for strategy '{strategy}'")
            return None
        
        # Get configuration for this strategy
        strategy_config = {}
        if self.config and "strategies" in self.config:
            strategy_config = self.config.get("strategies", {}).get(strategy, {})
        
        try:
            return retriever_class(
                vector_store=vector_store,
                embedder=embedder,
                **strategy_config
            )
        except Exception as e:
            logger.error(f"Error creating retriever '{strategy}': {str(e)}")
            return None
    
    def get_default_retriever(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder
    ) -> BaseRetriever:
        """Get the default retriever as specified in config."""
        default_strategy = "semantic"
        if self.config:
            default_strategy = self.config.get("default_strategy", default_strategy)
        
        retriever = self.get_retriever(default_strategy, vector_store, embedder)
        if not retriever:
            logger.warning(
                f"Default retriever '{default_strategy}' failed. "
                f"Falling back to 'semantic'."
            )
            retriever = self.get_retriever("semantic", vector_store, embedder)
            
            if not retriever:
                raise RuntimeError("Could not create any retriever")
        
        return retriever

