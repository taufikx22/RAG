import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai

from src.generation.base import BaseGenerator, GenerationResult

logger = logging.getLogger(__name__)


class GeminiGenerator(BaseGenerator):
    """Google Gemini API-based text generator."""
    
    def __init__(
        self,
        model: str = "gemini-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Gemini generator.
        
        Args:
            model: Gemini model name (e.g., 'gemini-pro', 'gemini-pro-vision')
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up API key
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not provided and GOOGLE_API_KEY not set"
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        try:
            self.model_instance = genai.GenerativeModel(model)
            logger.info(f"Initialized Gemini generator with model '{model}'")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{model}': {str(e)}")
            raise
        
        # Default system prompt
        self.system_prompt = system_prompt or """
You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to fully answer the question, say so clearly.
Be accurate, concise, and helpful.
"""
    
    def generate(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response using Gemini API."""
        start_time = datetime.now()
        
        # Use instance defaults if not provided
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            # Construct the prompt
            prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            # Make API call
            response = self.model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract response
            generated_text = response.text
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "generation_time": generation_time,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', None)
                },
                "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
                "timestamp": datetime.now().isoformat(),
                "provider": "Google Gemini"
            }
            
            # Log generation info
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_count = getattr(response.usage_metadata, 'candidates_token_count', 'unknown')
                logger.debug(
                    f"Generated response ({token_count} tokens) "
                    f"in {generation_time:.3f}s"
                )
            else:
                logger.debug(
                    f"Generated response in {generation_time:.3f}s"
                )
            
            return GenerationResult(
                query=query,
                response=generated_text,
                context=context,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        return {
            "provider": "Google",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "type": "api"
        }
