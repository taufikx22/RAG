from pathlib import Path
from typing import Dict, Type
import logging

from src.document_processing.base import DocumentProcessor
from src.document_processing.pdfProcessor import PDFProcessor
from src.document_processing.docxProcessor import DocxProcessor
from src.document_processing.htmlProcessor import HTMLProcessor

logger = logging.getLogger(__name__)

class DocumentProcessorFactory:
    def __init__(self):
        self._processors: Dict[str, Type[DocumentProcessor]] = {
            'pdf': PDFProcessor,
            'docx': DocxProcessor,
            'html': HTMLProcessor
        }

    def get_supported_extensions(self) -> list:
        return list(self._processors.keys())
    
    def get_processor(self, file_path: Path) -> DocumentProcessor:
        ext = file_path.suffix.lower().lstrip('.')
        processor_class = self._processors.get(ext)

        if processor_class is None:
            logger.error(f"No processor found for file extension: {ext}")
            raise ValueError(f"No processor found for file extension: {ext}")
            
    def process_file(self, file_path: Path) -> DocumentProcessor:
        processor = self.get_processor(file_path)
        if processor:
            return processor(file_path)
        else:
            logger.error(f"Failed to create processor for file: {file_path}")
            raise ValueError(f"Failed to create processor for file: {file_path}")
        
        try:
            result = processor.process(filr_path)
            return Document(
                content= result,
                doc_id= result.get('doc_id', None),
                metadata={
                    'file_name': file_path.name,
                    'file_extension': file_path.suffix
                }
            )
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise RuntimeError(f"Error processing file {file_path}: {e}") from e
        
        return processor_class(file_path)   
    