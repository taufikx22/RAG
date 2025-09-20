from typing import List, Dict, Any, Optional
import numpy as np
import logging
import time
from datetime import datetime
import os
from tqdm import tqdm
import openai

from src.chunking.base import Chunk
from src.embedding.base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API."""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        batch_size: int = 16,
        retry_limit: int = 3,
        retry_delay: float = 5.0,
        dimensions: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            batch_size: Batch size for embedding generation
            retry_limit: Number of retries on API failure
            retry_delay: Delay between retries in seconds
            dimensions: Override dimension of embeddings (model-dependent)
            show_progress: Whether to show progress bar during batch processing
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.batch_size = batch_size
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self._dimensions = dimensions  # Can be None, will be set on first API call
        self.show_progress = show_progress
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set."
            )
