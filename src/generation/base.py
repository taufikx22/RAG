from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Class representing a generation result."""
    
    query: str
    response: str
    context: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self) -> str:
        return f"GenerationResult(query='{self.query[:50]}...', response_len={len(self.response)})"


class BaseGenerator(ABC):
    """Base class for text generation."""
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        **kwargs
    ) -> GenerationResult:
        """
        Generate a response based on query and context.
        
        Args:
            query: The user query
            context: Retrieved context for augmentation
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result object
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the generation model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass
