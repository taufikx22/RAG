from typing import List, Dict, Any, Optional
import logging
import os
from google import genai

from src.chunking.base import Chunk
from src.embedding.base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)

class GeminiEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        batch_size: int = 16,
        retry_limit: int = 3,
        retry_delay: float = 5.0,
        dimensions: Optional[int] = None,
        show_progress: bool = True
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.batch_size = batch_size
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self._dimensions = dimensions
        self.show_progress = show_progress
        
        if not self.api_key:
            raise ValueError(
                "Google API key not provided and GOOGLE_API_KEY environment variable not set."
            )
      
    