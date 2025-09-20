import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import requests
import json

from src.generation.base import BaseGenerator, GenerationResult

logger = logging.getLogger(__name__)


class LocalLLMGenerator(BaseGenerator):
    """Local LLM generator using Ollama or similar local APIs."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize local LLM generator.
        
        Args:
            model: Local model name
            base_url: Base URL for the local LLM API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Default system prompt
        self.system_prompt = system_prompt or """
You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to fully answer the question, say so clearly.
Be accurate, concise, and helpful.
"""
        
        # Test connection
        self._test_connection()
        
        logger.info(f"Initialized local LLM generator with model '{model}'")
    
    def _test_connection(self) -> None:
        """Test connection to local LLM API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            logger.info("Successfully connected to local LLM API")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to local LLM API: {str(e)}")
    
    def generate(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response using local LLM."""
        start_time = datetime.now()
        
        # Use instance defaults if not provided
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            # Construct prompt
            prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            
            # Prepare request payload (Ollama format)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                },
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get("response", "")
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                "model": self.model,
                "provider": "local",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "generation_time": generation_time,
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(
                f"Generated response ({result.get('eval_count', 0)} tokens) "
                f"in {generation_time:.3f}s"
            )
            
            return GenerationResult(
                query=query,
                response=generated_text.strip(),
                context=context,
                metadata=metadata
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error with local LLM API request: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating response with local LLM: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local LLM model information."""
        return {
            "provider": "local",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "type": "local_api"
        }
