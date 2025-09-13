import uuid
from typing import List, Dict, Any, Optional
import logging
import re
import nltk

from src.document_processing.base import Document
from src.chunking.base import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        max_tokens: int = 1000,
        min_tokens: int = 100,
        separators: List[str] = None
    ):

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be less than max_tokens")
        
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.separators = separators or [
            # Headers
            "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
            # Double newlines (paragraphs)
            "\n\n",
            # Bullet points and numbered lists
            "\n- ", "\n* ", "\n+ ", "\n1. ", "\n2. ", "\n3. ",
            # Single newlines
            "\n",
            # Sentences 
            ". ", "! ", "? ",
            # Semicolons and colons
            "; ", ": ",
            # Commas
            ", "
        ]
        
    def _estimate_token_count(self, text: str) -> int:\
        return len(text.split())
    
    def _split_text(self, text: str, separator: str) -> List[str]:
        if not separator.strip(): 
            split_parts = text.split(separator)
            return [part + separator for part in split_parts[:-1]] + [split_parts[-1]]
        
        parts = []
        segments = text.split(separator)
        for i, segment in enumerate(segments):
            if i == 0:
                parts.append(segment)
            else:
                parts.append(separator + segment)
        return parts
    
    def _recursive_split(
        self, 
        text: str, 
        separator_idx: int = 0, 
        chunk_prefix: str = ""
    ) -> List[str]:
        
        if separator_idx >= len(self.separators):
            return [text]
        token_count = self._estimate_token_count(text)
        if token_count <= self.max_tokens:
            return [text]

        separator = self.separators[separator_idx]
        chunks = self._split_text(text, separator)
        if len(chunks) == 1:
            return self._recursive_split(text, separator_idx + 1, chunk_prefix)

        return self._process_chunks(chunks, separator_idx, chunk_prefix)

    def _process_chunks(
        self,
        chunks: List[str],
        separator_idx: int,
        chunk_prefix: str
    ) -> List[str]:
        result = []
        current_chunk = ""
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self._estimate_token_count(chunk)
            if chunk_tokens > self.max_tokens:
                self._finalize_current_chunk(result, current_chunk, current_tokens, chunk_prefix)
                current_chunk = ""
                current_tokens = 0
                sub_chunks = self._recursive_split(chunk, separator_idx + 1, chunk_prefix)
                result.extend(sub_chunks)
            elif current_tokens + chunk_tokens > self.max_tokens:
                if current_tokens >= self.min_tokens:
                    result.append(chunk_prefix + current_chunk)
                    current_chunk = chunk
                    current_tokens = chunk_tokens
                else:
                    current_chunk += chunk
                    current_tokens += chunk_tokens
                    if current_tokens > self.max_tokens * 1.5:
                        result.append(chunk_prefix + current_chunk)
                        current_chunk = ""
                        current_tokens = 0
            else:
                current_chunk += chunk
                current_tokens += chunk_tokens

        self._finalize_last_chunk(result, current_chunk, current_tokens, chunk_prefix)
        return result

    def _finalize_current_chunk(self, result, current_chunk, current_tokens, chunk_prefix):
        if current_chunk and current_tokens >= self.min_tokens:
            result.append(chunk_prefix + current_chunk)

    def _finalize_last_chunk(self, result, current_chunk, current_tokens, chunk_prefix):
        if current_chunk and current_tokens >= self.min_tokens:
            result.append(chunk_prefix + current_chunk)
        elif current_chunk:
            if result:
                result[-1] += current_chunk
            else:
                result.append(chunk_prefix + current_chunk)
        
    def chunk_document(self, document: Document) -> List[Chunk]:
        doc_content = document.content
        doc_id = document.doc_id or str(uuid.uuid4())
        
        text_chunks = self._recursive_split(doc_content)
        
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{i + 1}"
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
            ))
        
        return chunks
