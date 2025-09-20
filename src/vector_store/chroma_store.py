import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import chromadb
from chromadb.config import Settings

from src.vector_store.base import BaseVectorStore, VectorStoreResult
from src.embedding.base import EmbeddingResult
from src.chunking.base import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/chroma_db",
        distance_function: str = "cosine"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_function = distance_function
        
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_function}
            )
            
            logger.info(f"Initialized ChromaVectorStore with collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaVectorStore: {str(e)}")
            raise
    
    def add(self, embedding: List[EmbeddingResult]) -> None:
        self.add_embeddings(embedding)

    def add_embeddings(self, embeddings: List[EmbeddingResult]) -> None:
        try:
            if not embeddings:
                logger.warning("No embeddings provided to add")
                return
            
            ids = []
            vectors = []
            metadatas = []
            documents = []
            
            for emb in embeddings:
                ids.append(emb.chunk_id)
                vectors.append(emb.embedding.tolist())
                
                metadata = {}
                for key, value in emb.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = json.dumps(value)
                
                metadata['added_at'] = datetime.now().isoformat()
                metadatas.append(metadata)
                
                documents.append(
                    getattr(emb, 'chunk_text', emb.chunk_id)
                )
            
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to Chroma collection")
            
        except Exception as e:
            logger.error(f"Error adding embeddings to Chroma: {str(e)}")
            raise
    
    def _prepare_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        where_clause = {}
        for key, value in filters.items():
            where_clause[key] = {"$in": value} if isinstance(value, list) else {"$eq": value}
        return where_clause
    
    def _process_results(self, results) -> List[VectorStoreResult]:
        search_results = []
        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
    
        for i, chunk_id in enumerate(ids):
            distance = distances[i]
            score = 1.0 - distance if self.distance_function == "cosine" else 1.0 / (1.0 + distance)
            metadata = metadatas[i] if metadatas else {}
            document = documents[i] if documents else ""
            chunk = None
            if document and metadata:
                chunk = Chunk(
                    text=document,
                    doc_id=metadata.get('doc_id', ''),
                    chunk_id=chunk_id,
                    metadata=metadata
                )
            search_results.append(
                VectorStoreResult(
                    chunk_id=chunk_id,
                    score=score,
                    chunk=chunk,
                    metadata=metadata
                )
            )
        return search_results
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorStoreResult]:
        try:
            query_vector = query_embedding.tolist()
            where_clause = self._prepare_where_clause(filters)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            return self._process_results(results)
        except Exception as e:
            logger.error(f"Error searching Chroma: {str(e)}")
            raise
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks belonging to a document."""
        try:
            results = self.collection.get(
                where={"doc_id": {"$eq": document_id}},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.info(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id} from Chroma: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "distance_function": self.distance_function,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
