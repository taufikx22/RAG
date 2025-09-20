from typing import Dict, Type, List, Optional
import logging

from src.evaluation.base import BaseEvaluationMetric
from src.evaluation.basic_metrics import (
    ContextRelevanceMetric, AnswerRelevanceMetric,
    FaithfulnessMetric, AnswerAccuracyMetric
)
from src.evaluation.rag_evaluator import RAGEvaluator

logger = logging.getLogger(__name__)


class EvaluationFactory:
    """Factory for creating evaluation components."""
    
    def __init__(self, config=None):
        """
        Initialize evaluation factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._metrics: Dict[str, Type[BaseEvaluationMetric]] = {}
        
        # Register default metrics
        self.register_metric("context_relevance", ContextRelevanceMetric)
        self.register_metric("answer_relevance", AnswerRelevanceMetric)
        self.register_metric("faithfulness", FaithfulnessMetric)
        self.register_metric("answer_accuracy", AnswerAccuracyMetric)
    
    def register_metric(self, name: str, metric_class: Type[BaseEvaluationMetric]) -> None:
        """Register an evaluation metric class."""
        self._metrics[name.lower()] = metric_class
        logger.debug(f"Registered metric {metric_class.__name__} as '{name}'")
    
    def create_metric(self, metric_name: str, **kwargs) -> Optional[BaseEvaluationMetric]:
        """
        Create a metric instance.
        
        Args:
            metric_name: Name of the metric to create
            **kwargs: Additional arguments for metric initialization
            
        Returns:
            Metric instance or None
        """
        metric_name = metric_name.lower()
        metric_class = self._metrics.get(metric_name)
        
        if not metric_class:
            logger.error(f"No metric found for name '{metric_name}'")
            return None
        
        try:
            return metric_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating metric '{metric_name}': {str(e)}")
            return None
    
    def create_evaluator(
        self, 
        metric_names: Optional[List[str]] = None,
        **kwargs
    ) -> RAGEvaluator:
        """
        Create a RAG evaluator with specified metrics.
        
        Args:
            metric_names: List of metric names to include
            **kwargs: Additional arguments for evaluator initialization
            
        Returns:
            Configured RAG evaluator
        """
        # Use default metrics if none specified
        if metric_names is None:
            if self.config and "metrics" in self.config:
                metric_names = self.config["metrics"]
            else:
                metric_names = ["context_relevance", "answer_relevance", "faithfulness"]
        
        # Create metric instances
        metrics = []
        for metric_name in metric_names:
            metric = self.create_metric(metric_name)
            if metric:
                metrics.append(metric)
            else:
                logger.warning(f"Could not create metric '{metric_name}', skipping")
        
        if not metrics:
            logger.warning("No metrics available, using defaults")
            metrics = [
                ContextRelevanceMetric(),
                AnswerRelevanceMetric(),
                FaithfulnessMetric()
            ]
        
        # Get evaluator configuration
        evaluator_config = {}
        if self.config:
            evaluator_config.update({
                "save_results": self.config.get("logging_enabled", True),
                "results_dir": self.config.get("results_dir", "./data/evaluations")
            })
        evaluator_config.update(kwargs)
        
        return RAGEvaluator(metrics=metrics, **evaluator_config)
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names."""
        return list(self._metrics.keys())
