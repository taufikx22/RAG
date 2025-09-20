import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.base import (
    BaseEvaluationMetric, EvaluationResult, RAGEvaluationInput
)
from src.evaluation.basic_metrics import (
    ContextRelevanceMetric, AnswerRelevanceMetric,
    FaithfulnessMetric, AnswerAccuracyMetric
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Main class for evaluating RAG system performance."""

    def __init__(
        self,
        metrics: Optional[List[BaseEvaluationMetric]] = None,
        save_results: bool = True,
        results_dir: str = "./data/evaluations"
    ):
        """
        Initialize RAG evaluator.

        Args:
            metrics: List of evaluation metrics to use
            save_results: Whether to save evaluation results
            results_dir: Directory to save results
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)

        if metrics is None:
            self.metrics = [
                ContextRelevanceMetric(),
                AnswerRelevanceMetric(),
                FaithfulnessMetric(),
                AnswerAccuracyMetric()
            ]
        else:
            self.metrics = metrics

        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RAG evaluator with {len(self.metrics)} metrics")

    def evaluate_single(self, input_data: RAGEvaluationInput) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single query-answer pair.

        Args:
            input_data: Input data for evaluation

        Returns:
            Dictionary of metric name to evaluation result
        """
        results = {}

        for metric in self.metrics:
            try:
                if metric.requires_expected_answer and not input_data.expected_answer:
                    logger.debug(
                        f"Skipping {metric.name} - requires expected answer"
                    )
                    continue

                result = metric.evaluate(input_data)
                results[metric.name] = result

            except Exception as e:
                logger.error(f"Error evaluating {metric.name}: {str(e)}")
                results[metric.name] = EvaluationResult(
                    metric_name=metric.name,
                    score=0.0,
                    details={"error": str(e)}
                )

        return results

    def evaluate_batch(
        self,
        input_batch: List[RAGEvaluationInput],
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of query-answer pairs.

        Args:
            input_batch: List of input data for evaluation
            experiment_name: Optional name for this evaluation run

        Returns:
            Aggregated evaluation results
        """
        start_time = datetime.now()

        all_results = []
        for i, input_data in enumerate(input_batch):
            logger.debug(f"Evaluating sample {i+1}/{len(input_batch)}")
            sample_results = self.evaluate_single(input_data)

            result_entry = {
                "sample_id": i,
                "query": input_data.query,
                "has_expected_answer": input_data.expected_answer is not None,
                "has_context": input_data.retrieved_context is not None,
                "has_generated_answer": input_data.generated_answer is not None,
                "metadata": input_data.metadata,
                "metrics": {
                    name: result.to_dict()
                    for name, result in sample_results.items()
                }
            }
            all_results.append(result_entry)

        aggregated = self._aggregate_results(all_results)

        evaluation_time = (datetime.now() - start_time).total_seconds()
        aggregated["evaluation_metadata"] = {
            "experiment_name": experiment_name,
            "total_samples": len(input_batch),
            "evaluation_time": evaluation_time,
            "timestamp": datetime.now().isoformat(),
            "metrics_used": [metric.name for metric in self.metrics]
        }

        if self.save_results:
            self._save_results(aggregated, experiment_name)

        logger.info(
            f"Completed batch evaluation of {len(input_batch)} samples "
            f"in {evaluation_time:.2f}s"
        )

        return aggregated

    def _aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual evaluation results."""
        if not all_results:
            return {"error": "No results to aggregate"}

        metric_scores, _ = self._collect_metric_scores(all_results)
        aggregated_metrics = {
            metric_name: self._aggregate_metric_scores(scores)
            for metric_name, scores in metric_scores.items()
        }

        overall_score = (
            sum(metrics["mean"] for metrics in aggregated_metrics.values()) / len(aggregated_metrics)
            if aggregated_metrics else 0.0
        )

        return {
            "metrics": aggregated_metrics,
            "overall_score": overall_score
        }

    def _is_valid_score(self, score):
        import math
        return score is not None and not (isinstance(score, float) and math.isnan(score))  # Not NaN

    def _collect_metric_scores(self, all_results: List[Dict[str, Any]]):
        metric_scores = {}
        metric_counts = {}
        for result_entry in all_results:
            for metric_name, metric_data in result_entry.get("metrics", {}).items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                    metric_counts[metric_name] = 0
                score = metric_data.get("score", 0.0)
                if self._is_valid_score(score):
                    metric_scores[metric_name].append(score)
                    metric_counts[metric_name] += 1
        return metric_scores, metric_counts

    def _aggregate_metric_scores(self, scores):
        if scores:
            return {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
                "scores": scores
            }
        else:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "scores": []
            }

    def _save_results(
        self,
        results: Dict[str, Any],
        experiment_name: Optional[str] = None
    ) -> None:
        """Save evaluation results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if experiment_name:
                filename = f"{experiment_name}_{timestamp}.json"
            else:
                filename = f"evaluation_{timestamp}.json"

            filepath = self.results_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved evaluation results to {filepath}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")

    def add_metric(self, metric: BaseEvaluationMetric) -> None:
        """Add a new evaluation metric."""
        self.metrics.append(metric)
        logger.info(f"Added evaluation metric: {metric.name}")

    def get_metric_names(self) -> List[str]:
        """Get names of all configured metrics."""
        return [metric.name for metric in self.metrics]

