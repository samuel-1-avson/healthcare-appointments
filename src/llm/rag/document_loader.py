# src/llm/rag/document_loader.py
"""
Document Loading
================
Load and process documents for RAG.

Supports:
- Markdown files
- Text files
- PDF files (with optional dependencies)
- Web pages
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)

logger = logging.getLogger(__name__)


# ==================== Document Metadata ====================

@dataclass
class DocumentMetadata:
    """Metadata for a loaded document."""
    
    source: str
    filename: str
    file_type: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    document_id: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate document ID if not provided."""
        if not self.document_id:
            self.document_id = hashlib.md5(
                f"{self.source}:{self.filename}".encode()
            ).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "filename": self.filename,
            "file_type": self.file_type,
            "title": self.title,
            "document_id": self.document_id,
            "word_count": self.word_count,
            "char_count": self.char_count,
            **self.custom_metadata
        }


# ==================== Document Loader ====================

class DocumentLoader:
    """
    Load documents from various sources for RAG.
    
    Supports:
    - Single files (markdown, text)
    - Directories
    - Multiple file types
    
    Example
    -------
    >>> loader = DocumentLoader()
    >>> docs = loader.load_directory("data/documents")
    >>> print(f"Loaded {len(docs)} documents")
    """
    
    SUPPORTED_EXTENSIONS = {
        ".md": "markdown",
        ".txt": "text",
        ".pdf": "pdf",
        ".html": "html"
    }
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        extract_metadata: bool = True
    ):
        """
        Initialize the document loader.
        
        Parameters
        ----------
        base_path : str, optional
            Base path for document loading
        extract_metadata : bool
            Whether to extract metadata from documents
        """
        self.base_path = Path(base_path) if base_path else Path("data/documents")
        self.extract_metadata = extract_metadata
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Tracking
        self._loaded_docs: List[Document] = []
        self._load_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {}
        }
    
    def load_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load a single file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to file
        metadata : dict, optional
            Additional metadata to attach
        
        Returns
        -------
        List[Document]
            Loaded documents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        
        self.logger.info(f"Loading file: {file_path}")
        
        try:
            # Choose loader based on file type
            if extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif extension == ".txt":
                loader = TextLoader(str(file_path))
            else:
                # Fallback to text loader
                loader = TextLoader(str(file_path))
            
            docs = loader.load()
            
            # Enhance metadata
            for doc in docs:
                doc_metadata = self._create_metadata(file_path, doc.page_content)
                doc.metadata.update(doc_metadata.to_dict())
                
                if metadata:
                    doc.metadata.update(metadata)
            
            self._update_stats(extension, success=True)
            self._loaded_docs.extend(docs)
            
            self.logger.info(f"Loaded {len(docs)} document(s) from {file_path.name}")
            return docs
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            self._update_stats(extension, success=False)
            raise
    
    def load_directory(
        self,
        directory: Optional[Union[str, Path]] = None,
        glob_pattern: str = "**/*",
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Parameters
        ----------
        directory : str or Path, optional
            Directory path (default: base_path)
        glob_pattern : str
            Glob pattern for file matching
        recursive : bool
            Search subdirectories
        
        Returns
        -------
        List[Document]
            All loaded documents
        """
        directory = Path(directory) if directory else self.base_path
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        self.logger.info(f"Loading documents from: {directory}")
        
        all_docs = []
        
        # Find all matching files
        pattern = glob_pattern if recursive else f"*"
        
        for extension in self.SUPPORTED_EXTENSIONS.keys():
            files = list(directory.glob(f"{pattern}{extension}"))
            
            for file_path in files:
                try:
                    docs = self.load_file(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    self.logger.warning(f"Skipping {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(all_docs)} total documents")
        return all_docs
    
    def load_multiple(
        self,
        file_paths: List[Union[str, Path]]
    ) -> List[Document]:
        """Load multiple specific files."""
        all_docs = []
        
        for path in file_paths:
            try:
                docs = self.load_file(path)
                all_docs.extend(docs)
            except Exception as e:
                self.logger.warning(f"Failed to load {path}: {e}")
        
        return all_docs
    
    def _create_metadata(
        self,
        file_path: Path,
        content: str
    ) -> DocumentMetadata:
        """Create metadata for a document."""
        # Extract title from content (first heading)
        title = None
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        return DocumentMetadata(
            source=str(file_path),
            filename=file_path.name,
            file_type=self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "unknown"),
            title=title or file_path.stem,
            word_count=len(content.split()),
            char_count=len(content)
        )
    
    def _update_stats(self, extension: str, success: bool):
        """Update loading statistics."""
        self._load_stats["total_files"] += 1
        
        if success:
            self._load_stats["successful"] += 1
        else:
            self._load_stats["failed"] += 1
        
        file_type = self.SUPPORTED_EXTENSIONS.get(extension, "unknown")
        self._load_stats["by_type"][file_type] = \
            self._load_stats["by_type"].get(file_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            **self._load_stats,
            "total_documents": len(self._loaded_docs)
        }
    
    def get_loaded_documents(self) -> List[Document]:
        """Get all loaded documents."""
        return self._loaded_docs
    
    def clear(self):
        """Clear loaded documents and reset stats."""
        self._loaded_docs = []
        self._load_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {}
        }


# ==================== Convenience Functions ====================

def load_document(file_path: Union[str, Path]) -> List[Document]:
    """Load a single document."""
    loader = DocumentLoader()
    return loader.load_file(file_path)


def load_policy_documents(
    directory: str = "data/documents"
) -> List[Document]:
    """
    Load all policy documents from the default directory.
    
    Parameters
    ----------
    directory : str
        Path to documents directory
    
    Returns
    -------
    List[Document]
        All policy documents
    """
    loader = DocumentLoader(base_path=directory)
    docs = loader.load_directory()
    
    logger.info(f"Loaded {len(docs)} policy documents")
    logger.info(f"Stats: {loader.get_stats()}")
    
    return docs