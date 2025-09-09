import PyPDF2
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime
import hashlib

from src.document_processing.base import DocumentProcessor

logger = logging.getLogger(__name__)

class PDFProcessor(DocumentProcessor):
    def process(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"The file {file_path} is not a PDF document.")
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                content = ""
                for page_num in range(len(reader.pages)):
                    page= reader.pages[page_num]
                    content += page.extract_text() or ""

                metadata = {
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'page_count': len(reader.pages),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }

                if reader.metadata:
                    for key, value in reader.metadata.items():
                        if key.startswith('/'):
                            key = key[1:]
                            metadata[key] = value

                doc_id = hashlib.md5(file_path.name+ content.encode()).hexdigest()
                metadata['doc_id'] = doc_id
                return {
                    'content': content,
                    'metadata': metadata,
                    'doc_id': doc_id
                }
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            raise RuntimeError(f"Failed to process PDF file {file_path}.") from e
        
