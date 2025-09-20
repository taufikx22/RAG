from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class EvaluationResult:
    """Class representing an evaluation result."""
    
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details
        }


@dataclass
class RAGEvaluationInput:
    """Input for RAG evaluation."""
    
    query: str
    expected_answer: Optional[str] = None
    retrieved_context: Optional[str] = None
    generated_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseEvaluationMetric(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """
        Evaluate the given input.
        
        Args:
            input_data: Input data for evaluation
            
        Returns:
            Evaluation result
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        pass
    
    @property
    @abstractmethod
    def requires_expected_answer(self) -> bool:
        """Return True if this metric requires an expected answer."""
        pass

