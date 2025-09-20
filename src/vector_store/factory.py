from typing import Dict, Type, Optional
import logging

from src.vector_store.base import BaseVectorStore
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    def __init__(self, config=None):
        """
        Initialize vector store factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._stores: Dict[str, Type[BaseVectorStore]] = {}
        
        # Register default stores
        self.register_store("chroma", ChromaVectorStore)
        self.register_store("qdrant", QdrantVectorStore)
    
    def register_store(self, name: str, store_class: Type[BaseVectorStore]) -> None:
        """Register a vector store class."""
        self._stores[name.lower()] = store_class
        logger.debug(f"Registered vector store {store_class.__name__} as '{name}'")
    
    def get_store(self, store_type: str) -> Optional[BaseVectorStore]:
        """
        Get a vector store instance.
        
        Args:
            store_type: Type of vector store to create
            
        Returns:
            Configured vector store instance or None
        """
        store_type = store_type.lower()
        store_class = self._stores.get(store_type)
        
        if not store_class:
            logger.error(f"No vector store found for type '{store_type}'")
            return None
        
        # Get configuration for this store type
        store_config = {}
        if self.config and "options" in self.config:
            store_config = self.config.get("options", {}).get(store_type, {})
        
        try:
            return store_class(**store_config)
        except Exception as e:
            logger.error(f"Error creating vector store '{store_type}': {str(e)}")
            return None
    
    def get_default_store(self) -> BaseVectorStore:
        """Get the default vector store as specified in config."""
        default_type = "chroma"
        if self.config:
            default_type = self.config.get("default", default_type)
        
        store = self.get_store(default_type)
        if not store:
            logger.warning(f"Default store '{default_type}' failed. Falling back to 'chroma'.")
            store = self.get_store("chroma")
            
            if not store:
                raise RuntimeError("Could not create any vector store")
        
        return store
