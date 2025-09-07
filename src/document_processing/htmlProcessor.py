from bs4 import BeautifulSoup
from typing import List, Dict, Any
from pathlib import Path
import html2text
import logging
from datetime import datetime
import hashlib

from src.document_processing.base import DocumentProcessor

class HTMLProcessor(DocumentProcessor):
    def __init__(self, file_path: str):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0
        self.file_path = Path(file_path)

    def process(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if file_path.suffix.lower() != '.html':
            raise ValueError(f"The file {file_path} is not an HTML file.")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            soup = BeautifulSoup(content, 'html.parser')

            metadata = {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': 'HTML',
                'processed_at' : datetime.now().isoformat(),
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'hash': hashlib.sha256(content.encode('utf-8')).hexdigest()
            }

            if soup.title:
                metadata['title'] = soup.title.string.strip()
            
            for meta in soup.find_all('meta'):
                if 'name' in meta.attrs and meta['name'].lower() == 'description':
                    metadata['description'] = meta.get('content', '').strip()
                elif 'property' in meta.attrs and meta['property'].lower() == 'og:description':
                    metadata['og_description'] = meta.get('content', '').strip()

            content= self.html_converter.handle(str(soup))

            doc_id= hashlib.md5(content.encode('utf-8')).hexdigest()
            metadata['doc_id'] = doc_id

            return {
                'content': content,
                'metadata': metadata,
                'doc_id': doc_id
            }
        
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise RuntimeError(f"Failed to process HTML file: {e}") from e
