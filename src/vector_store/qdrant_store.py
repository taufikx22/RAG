import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

from src.vector_store.base import BaseVectorStore, VectorStoreResult
from src.embedding.base import EmbeddingResult
from src.chunking.base import Chunk

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str = "rag_documents",
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "Cosine"
    ):
        self.collection_name = collection_name
        self.url = url
        self.vector_size = vector_size

        try:
            self.client = QdrantClient(
                url=url,
                api_key=api_key
            )

            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Manhattan": Distance.MANHATTAN
            }
            self.distance = distance_map.get(distance, Distance.COSINE)
            self._ensure_collection_exists()

            logger.info(f"Initialized QdrantVectorStore with collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantVectorStore: {str(e)}")
            raise

    def _ensure_collection_exists(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}'")
            else:
                logger.info(f"Using existing Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def add_embeddings(self, embeddings: List[EmbeddingResult]) -> None:
        try:
            if not embeddings:
                logger.warning("No embeddings provided to add")
                return

            points = []

            for emb in embeddings:
                payload = dict(emb.metadata)
                payload['chunk_id'] = emb.chunk_id
                payload['added_at'] = datetime.now().isoformat()

                vector = emb.embedding
                if len(vector) != self.vector_size:
                    logger.warning(
                        f"Vector size mismatch: expected {self.vector_size}, "
                        f"got {len(vector)} for chunk {emb.chunk_id}"
                    )
                    continue

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload=payload
                )
                points.append(point)

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Added {len(points)} embeddings to Qdrant collection")
        except Exception as e:
            logger.error(f"Error adding embeddings to Qdrant: {str(e)}")
            raise

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Helper to build Qdrant filter from filters dict."""
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.extend(
                    FieldCondition(key=key, match=MatchValue(value=v)) for v in value
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions) if conditions else None

    def _process_search_results(self, search_results) -> List[VectorStoreResult]:
        """Helper to process Qdrant search results."""
        results = []
        for res in search_results:
            chunk_id = res.payload.get('chunk_id', str(res.id))
            score = res.score
            chunk = None
            if 'doc_id' in res.payload:
                chunk = Chunk(
                    text=res.payload.get('text', ''),
                    doc_id=res.payload['doc_id'],
                    chunk_id=chunk_id,
                    metadata=res.payload
                )
            results.append(VectorStoreResult(
                chunk_id=chunk_id,
                score=score,
                chunk=chunk,
                metadata=res.payload
            ))
        return results

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorStoreResult]:
        try:
            filter_obj = self._build_filter(filters)
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=filter_obj,
                with_payload=True
            )
            results = self._process_search_results(search_results)
            logger.debug(f"Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"Error searching Qdrant collection: {str(e)}")
            raise

    def delete_by_document_id(self, doc_id: str) -> bool:
        try:
            filter_obj = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=None,
                limit=10000,
                query_filter=filter_obj,
                with_payload=True
            )

            results = []
            for res in search_results:
                chunk_id = res.payload.get('chunk_id', str(res.id))
                score = res.score
                chunk = None
                if 'doc_id' in res.payload:
                    chunk = Chunk(
                        text=res.payload.get('text', ''),
                        doc_id=res.payload['doc_id'],
                        chunk_id=chunk_id,
                        metadata=res.payload
                    )
                results.append(VectorStoreResult(
                    chunk_id=chunk_id,
                    score=score,
                    chunk=chunk,
                    metadata=res.payload
                ))

            logger.debug(f"Found {len(results)} results for deletion")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_obj
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting by document ID: {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": info.points_count,
                "vector_size": self.vector_size,
                "distance": self.distance.name,
                "url": self.url
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}

