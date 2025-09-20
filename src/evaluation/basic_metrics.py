import logging
import re
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.base import BaseEvaluationMetric, EvaluationResult, RAGEvaluationInput

logger = logging.getLogger(__name__)


class ContextRelevanceMetric(BaseEvaluationMetric):
    """Evaluate how relevant the retrieved context is to the query."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize context relevance metric.
        
        Args:
            threshold: Minimum relevance threshold
        """
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "context_relevance"
    
    @property
    def requires_expected_answer(self) -> bool:
        return False
    
    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """Evaluate context relevance using TF-IDF similarity."""
        try:
            if not input_data.retrieved_context or not input_data.query:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    details={"error": "Missing context or query"}
                )
            
            # Calculate TF-IDF similarity between query and context
            texts = [input_data.query, input_data.retrieved_context]
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, similarity))
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "meets_threshold": score >= self.threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating context relevance: {str(e)}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )


class AnswerRelevanceMetric(BaseEvaluationMetric):
    """Evaluate how relevant the generated answer is to the query."""
    
    def __init__(self, threshold: float = 0.2):
        """
        Initialize answer relevance metric.
        
        Args:
            threshold: Minimum relevance threshold
        """
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "answer_relevance"
    
    @property
    def requires_expected_answer(self) -> bool:
        return False
    
    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """Evaluate answer relevance using TF-IDF similarity."""
        try:
            if not input_data.generated_answer or not input_data.query:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    details={"error": "Missing answer or query"}
                )
            
            # Calculate TF-IDF similarity between query and answer
            texts = [input_data.query, input_data.generated_answer]
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except ValueError:  # Handle case where vocabulary is empty
                similarity = 0.0
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, similarity))
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "meets_threshold": score >= self.threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )


class FaithfulnessMetric(BaseEvaluationMetric):
    """Evaluate how faithful the generated answer is to the retrieved context."""
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize faithfulness metric.
        
        Args:
            threshold: Minimum faithfulness threshold
        """
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "faithfulness"
    
    @property
    def requires_expected_answer(self) -> bool:
        return False
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text (simplified approach)."""
        # Split by sentences and filter out questions and very short sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 10 and 
                not sentence.endswith('?') and 
                not sentence.startswith(('However', 'But', 'Although'))):
                claims.append(sentence)
        
        return claims
    
    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """Evaluate faithfulness by checking if answer claims are supported by context."""
        try:
            if not input_data.generated_answer or not input_data.retrieved_context:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    details={"error": "Missing answer or context"}
                )
            
            # Extract claims from the generated answer
            answer_claims = self._extract_claims(input_data.generated_answer)
            
            if not answer_claims:
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,  # No claims to verify
                    details={"claims": [], "supported_claims": []}
                )
            
            # Check how many claims are supported by the context
            supported_claims = []
            context_lower = input_data.retrieved_context.lower()
            
            for claim in answer_claims:
                claim_lower = claim.lower()
                
                # Simple keyword overlap check (can be improved with semantic similarity)
                claim_words = set(re.findall(r'\b\w+\b', claim_lower))
                context_words = set(re.findall(r'\b\w+\b', context_lower))
                
                # Calculate word overlap
                overlap = len(claim_words.intersection(context_words))
                overlap_ratio = overlap / len(claim_words) if claim_words else 0
                
                if overlap_ratio > 0.3:  # At least 30% word overlap
                    supported_claims.append(claim)
            
            # Calculate faithfulness score
            score = len(supported_claims) / len(answer_claims) if answer_claims else 1.0
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                details={
                    "total_claims": len(answer_claims),
                    "supported_claims_count": len(supported_claims),
                    "claims": answer_claims,
                    "supported_claims": supported_claims,
                    "threshold": self.threshold,
                    "meets_threshold": score >= self.threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {str(e)}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )


class AnswerAccuracyMetric(BaseEvaluationMetric):
    """Evaluate accuracy of generated answer against expected answer."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize answer accuracy metric.
        
        Args:
            threshold: Minimum accuracy threshold
        """
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "answer_accuracy"
    
    @property
    def requires_expected_answer(self) -> bool:
        return True
    
    def evaluate(self, input_data: RAGEvaluationInput) -> EvaluationResult:
        """Evaluate answer accuracy using semantic similarity."""
        try:
            if not input_data.generated_answer or not input_data.expected_answer:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    details={"error": "Missing generated or expected answer"}
                )
            
            # Calculate TF-IDF similarity between generated and expected answers
            texts = [input_data.generated_answer, input_data.expected_answer]
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except ValueError:
                similarity = 0.0
            
            # Normalize score to 0-1 range
            score = max(0.0, min(1.0, similarity))
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                details={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "meets_threshold": score >= self.threshold,
                    "generated_length": len(input_data.generated_answer),
                    "expected_length": len(input_data.expected_answer)
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating answer accuracy: {str(e)}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )

