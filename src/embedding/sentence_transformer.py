from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


from src.chunking.base import Chunk
from src.embedding.base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using Sentence Transformers models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        show_progress: bool = True
    ):
        """
        Initialize the Sentence Transformer embedder.
        
        Args:
            model_name: Name of the model to use
            batch_size: Batch size for embedding generation
            device: Device to use (cpu, cuda, cuda:0, etc.)
            normalize_embeddings: Whether to normalize embeddings
            show_progress: Whether to show progress bar during batch processing
        """
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            self.batch_size = batch_size
            self.normalize_embeddings = normalize_embeddings
            self.show_progress = show_progress
            logger.info(f"Initialized SentenceTransformerEmbedder with model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformerEmbedder: {str(e)}")
            raise
    
    def embed_chunk(self, chunk: Chunk) -> EmbeddingResult:
        """Generate embedding for a single document chunk."""
        try:
            start_time = datetime.now()
            
            # Generate embedding
            embedding = self.model.encode(
                [chunk.text],
                batch_size=1,
                normalize_embeddings=self.normalize_embeddings
            )[0]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                **chunk.metadata,
                "model": self.model_name,
                "processing_time": processing_time,
                "embedding_timestamp": datetime.now().isoformat()
            }
            
            return EmbeddingResult(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error embedding chunk {chunk.chunk_id}: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple document chunks in batches."""
        try:
            start_time = datetime.now()
            
            # Extract text from chunks
            texts = [chunk.text for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=self.show_progress
            )
            
            # Calculate processing time
            total_time = (datetime.now() - start_time).total_seconds()
            avg_time = total_time / len(chunks) if chunks else 0
            
            # Create embedding results
            results = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    **chunk.metadata,
                    "model": self.model_name,
                    "batch_index": i,
                    "total_batch_time": total_time,
                    "avg_time_per_chunk": avg_time,
                    "embedding_timestamp": datetime.now().isoformat()
                }
                
                results.append(EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding,
                    metadata=metadata
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error embedding batch of {len(chunks)} chunks: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a search query."""
        try:
            return self.model.encode(
                query,
                normalize_embeddings=self.normalize_embeddings
            )
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        # This is a simple way to check the dimension - encode a dummy string
        return len(self.model.encode("dimension check"))
