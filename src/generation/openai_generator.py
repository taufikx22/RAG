import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from openai import OpenAI

from src.generation.base import BaseGenerator, GenerationResult

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseGenerator):
    """OpenAI API-based text generator."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize OpenAI generator.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up API key
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY not set"
            )
        
        # Initialize client
        self.client = OpenAI(api_key=api_key)
        
        # Default system prompt
        self.system_prompt = system_prompt or """
You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to fully answer the question, say so clearly.
Be accurate, concise, and helpful.
"""
        
        logger.info(f"Initialized OpenAI generator with model '{model}'")
    
    def generate(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response using OpenAI API."""
        start_time = datetime.now()
        
        # Use instance defaults if not provided
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            # Construct messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response
            generated_text = response.choices[0].message.content
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "generation_time": generation_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(
                f"Generated response ({response.usage.completion_tokens} tokens) "
                f"in {generation_time:.3f}s"
            )
            
            return GenerationResult(
                query=query,
                response=generated_text,
                context=context,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "type": "api"
        }

