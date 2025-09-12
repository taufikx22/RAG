import uuid
from typing import List, Optional
import logging

from src.document_processing.base import Document
from src.chunking.base import BaseChunker, Chunk

logger= logging.getLogger(__name__)

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int= 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        doc_content = document.content
        doc_id = document.doc_id or str(uuid.uuid4())

        if len(doc_content) <= self.chunk_size:
            return [Chunk(
                text=doc_content,
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4()),
                metadata={**(document.metadata or {}), 'chunk_index': 0, 'total_chunks': 1}
            )]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(doc_content):
            end = start + self.chunk_size

            if chunks and end < len(doc_content):
                end = self._find_natural_end(doc_content, start, end)

            chunk_text = doc_content[start:end]

            chunk_id = f"{doc_id}_chunk_{chunk_index+1}"
            chunk = Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata={**(document.metadata or {}), 'chunk_index': chunk_index, 'total_chunks': -1}
            )

            chunks.append(chunk)

            start = end - self.chunk_overlap
            chunk_index += 1

        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks

    def _find_natural_end(self, doc_content: str, start: int, end: int) -> int:
        natural_end = doc_content.rfind('\n', start, end)
        if natural_end == -1 or natural_end <= start:
            natural_end = doc_content.find(', ', start, end)
        if natural_end == -1 or natural_end <= start:
            natural_end = doc_content.find('  ', start, end)
        if natural_end > start:
            return natural_end
        return end