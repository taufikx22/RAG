import uuid
from typing import List, Dict, Any, Optional
import logging
import nltk
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from src.document_processing.base import Document
from src.chunking.base import BaseChunker, Chunk

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    def __init__(
        self,
        min_chunk_size: int = 256,
        max_chunk_size: int = 1024,
        paragraph_separator: str = "\n\n",
        respect_sentences: bool = True
    ):
        if min_chunk_size <= 0 or max_chunk_size <= 0:
            raise ValueError("Chunk sizes must be greater than 0.")
        if min_chunk_size > max_chunk_size:
            raise ValueError("Minimum chunk size cannot be greater than maximum chunk size.")
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.paragraph_separator = paragraph_separator
        self.respect_sentences = respect_sentences

    def chunk_document(self, document: Document) -> List[Chunk]:
        doc_content = document.content
        doc_id = document.doc_id or str(uuid.uuid4())

        if len(doc_content) <= self.max_chunk_size:
            return [self._create_chunk(doc_content, doc_id, 0, 1, document.metadata)]

        paragraphs = self._split_paragraphs(doc_content)
        chunks = self._process_paragraphs(paragraphs, doc_id, document.metadata)
        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)
        return chunks

    def _split_paragraphs(self, doc_content: str) -> List[str]:
        paragraphs = re.split(self.paragraph_separator, doc_content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _create_chunk(self, text: str, doc_id: str, chunk_index: int, total_chunks: int, metadata: Optional[dict], extra_metadata: Optional[dict] = None) -> Chunk:
        meta = {**(metadata or {}), 'chunk_index': chunk_index, 'total_chunks': total_chunks}
        if extra_metadata:
            meta.update(extra_metadata)
        return Chunk(
            text=text,
            doc_id=doc_id,
            chunk_id=f"{doc_id}_chunk_{chunk_index + 1}",
            metadata=meta
        )

    def _process_paragraphs(self, paragraphs: List[str], doc_id: str, metadata: Optional[dict]) -> List[Chunk]:
        chunks = []
        current_chunk_text = ""
        current_chunk_paragraphs = []
        chunk_index = 0

        for i, paragraph in enumerate(paragraphs):
            if self._should_split_chunk(current_chunk_text, paragraph):
                chunks.append(self._create_chunk(
                    current_chunk_text, doc_id, chunk_index, 0, metadata,
                    {'paragraphs': current_chunk_paragraphs}
                ))
                current_chunk_text = ""
                current_chunk_paragraphs = []
                chunk_index += 1

            if len(paragraph) > self.max_chunk_size:
                sentence_chunks, chunk_index = self._process_sentences(
                    paragraph, doc_id, metadata, chunk_index, i
                )
                chunks.extend(sentence_chunks)
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n"
                current_chunk_text += paragraph
                current_chunk_paragraphs.append(i)

        if current_chunk_text:
            chunks.append(self._create_chunk(
                current_chunk_text, doc_id, chunk_index, 0, metadata,
                {'paragraphs': current_chunk_paragraphs}
            ))
        return chunks

    def _should_split_chunk(self, current_chunk_text: str, paragraph: str) -> bool:
        return (len(current_chunk_text) + len(paragraph) > self.max_chunk_size and
                len(current_chunk_text) >= self.min_chunk_size)

    def _process_sentences(self, paragraph: str, doc_id: str, metadata: Optional[dict], chunk_index: int, paragraph_index: int):
        sentences = sent_tokenize(paragraph)
        sentence_chunk_text = ""
        sentence_indices = []
        sentence_chunks = []

        for j, sentence in enumerate(sentences):
            if (len(sentence_chunk_text) + len(sentence) > self.max_chunk_size and
                len(sentence_chunk_text) >= self.min_chunk_size):
                sentence_chunks.append(self._create_chunk(
                    sentence_chunk_text, doc_id, chunk_index, 0, metadata,
                    {'paragraphs': sentence_indices}
                ))
                sentence_chunk_text = ""
                sentence_indices = []
                chunk_index += 1
            if sentence_chunk_text:
                sentence_chunk_text += " "
            sentence_chunk_text += sentence
            sentence_indices.append(j)

        if sentence_chunk_text:
            sentence_chunks.append(self._create_chunk(
                sentence_chunk_text, doc_id, chunk_index, 0, metadata,
                {'paragraph_index': paragraph_index, 'sentence_indices': sentence_indices}
            ))
            chunk_index += 1
        return sentence_chunks, chunk_index
