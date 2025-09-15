from typing import Dict, Type, Optional
import logging

from src.document_processing.base import Document
from src.chunking.base import BaseChunker, Chunk
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.recursive_chunker import RecursiveChunker

logger = logging.getLogger(__name__)


class ChunkerFactory:
    
    def __init__(self, config=None):
        self.config = config or {}
        self._chunkers: Dict[str, Type[BaseChunker]] = {}
        
        self.register_chunker("fixed", FixedSizeChunker)
        self.register_chunker("semantic", SemanticChunker)
        self.register_chunker("recursive", RecursiveChunker)
        
    def register_chunker(self, name: str, chunker_class: Type[BaseChunker]) -> None:
        self._chunkers[name.lower()] = chunker_class
        logger.debug(f"Registered chunker {chunker_class.__name__} as '{name}'")
    
    def get_chunker(self, strategy: str) -> Optional[BaseChunker]:
        strategy = strategy.lower()
        chunker_class = self._chunkers.get(strategy)
        
        if not chunker_class:
            logger.warning(f"No chunker found for strategy '{strategy}'")
            return None
        
        strategy_config = {}
        if self.config and "strategies" in self.config:
            strategy_config = self.config.get("strategies", {}).get(strategy, {})
        
        try:
            return chunker_class(**strategy_config)
        except Exception as e:
            logger.error(f"Error creating chunker for strategy '{strategy}': {str(e)}")
            return chunker_class()
    
    def get_default_chunker(self) -> BaseChunker:
        default_strategy = "semantic"
        if self.config:
            default_strategy = self.config.get("default_strategy", default_strategy)
        
        chunker = self.get_chunker(default_strategy)
        if not chunker:
            logger.warning(
                f"Default chunker '{default_strategy}' not found. "
                f"Falling back to 'semantic'."
            )
            chunker = self.get_chunker("semantic")
            if not chunker:
                logger.warning("Falling back to 'fixed' chunker.")
                chunker = FixedSizeChunker()
                
        return chunker
    
    def chunk_document(self, document: Document, strategy: Optional[str] = None) -> list[Chunk]:
        if strategy:
            chunker = self.get_chunker(strategy)
            if not chunker:
                logger.warning(
                    f"Chunker '{strategy}' not found, using default instead."
                )
                chunker = self.get_default_chunker()
        else:
            chunker = self.get_default_chunker()
            
        return chunker.chunk_document(document)