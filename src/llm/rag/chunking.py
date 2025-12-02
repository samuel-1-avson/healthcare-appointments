# src/llm/rag/chunking.py
"""
Text Chunking
=============
Split documents into chunks for embedding and retrieval.

Chunking strategies:
- Fixed size with overlap
- Recursive (respects document structure)
- Semantic (based on content)
- Markdown-aware (preserves headings)
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # Markdown-specific
    preserve_headings: bool = True
    heading_levels: List[int] = None  # [1, 2, 3] for h1, h2, h3
    
    # Metadata
    add_chunk_metadata: bool = True
    include_source_in_chunk: bool = False
    
    def __post_init__(self):
        if self.heading_levels is None:
            self.heading_levels = [1, 2, 3]


class TextChunker:
    """
    Split documents into chunks for RAG.
    
    Supports multiple strategies and preserves document structure.
    
    Example
    -------
    >>> chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    >>> chunks = chunker.chunk_documents(documents)
    >>> print(f"Created {len(chunks)} chunks")
    """
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        """
        Initialize the chunker.
        
        Parameters
        ----------
        config : ChunkingConfig, optional
            Full configuration object
        chunk_size : int
            Target chunk size in characters
        chunk_overlap : int
            Overlap between chunks
        strategy : ChunkingStrategy
            Chunking strategy to use
        """
        if config:
            self.config = config
        else:
            self.config = ChunkingConfig(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._splitter = self._create_splitter()
        
        # Statistics
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "avg_chunk_size": 0
        }
    
    def _create_splitter(self):
        """Create the appropriate text splitter."""
        if self.config.strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        elif self.config.strategy == ChunkingStrategy.RECURSIVE:
            # Markdown-aware separators
            separators = [
                "\n## ",      # H2 headers
                "\n### ",     # H3 headers
                "\n#### ",    # H4 headers
                "\n\n",       # Paragraphs
                "\n",         # Lines
                ". ",         # Sentences
                " ",          # Words
                ""            # Characters
            ]
            
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=separators,
                length_function=len
            )
        
        else:  # FIXED
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
    
    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Chunk a list of documents.
        
        Parameters
        ----------
        documents : List[Document]
            Documents to chunk
        
        Returns
        -------
        List[Document]
            Chunked documents with metadata
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
        )
        
        return all_chunks
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document.
        
        Parameters
        ----------
        document : Document
            Document to chunk
        
        Returns
        -------
        List[Document]
            Chunked documents
        """
        self._stats["documents_processed"] += 1
        
        # Extract section context if markdown
        section_info = self._extract_sections(document.page_content)
        
        # Split the document
        chunks = self._splitter.split_documents([document])
        
        # Enhance chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_count"] = len(chunks)
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            # Add section context
            section = self._find_chunk_section(
                chunk.page_content, 
                section_info
            )
            if section:
                chunk.metadata["section"] = section
            
            # Add header context if preserving
            if self.config.preserve_headings:
                header = self._extract_nearest_header(
                    document.page_content,
                    chunk.page_content
                )
                if header:
                    chunk.metadata["header"] = header
        
        self._stats["chunks_created"] += len(chunks)
        
        return chunks
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract section information from markdown."""
        sections = []
        current_section = {"level": 0, "title": "Introduction", "start": 0}
        
        for match in re.finditer(r'^(#{1,4})\s+(.+)$', content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            sections.append({
                "level": level,
                "title": title,
                "start": match.start(),
                "end": match.end()
            })
        
        return sections
    
    def _find_chunk_section(
        self,
        chunk_content: str,
        sections: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Find which section a chunk belongs to."""
        for section in sections:
            if section["title"].lower() in chunk_content.lower():
                return section["title"]
        return None
    
    def _extract_nearest_header(
        self,
        full_content: str,
        chunk_content: str
    ) -> Optional[str]:
        """Extract the nearest header before this chunk."""
        chunk_start = full_content.find(chunk_content)
        
        if chunk_start == -1:
            return None
        
        # Search backwards for headers
        content_before = full_content[:chunk_start]
        
        headers = re.findall(r'^(#{1,4})\s+(.+)$', content_before, re.MULTILINE)
        
        if headers:
            return headers[-1][1].strip()
        
        return None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk plain text.
        
        Parameters
        ----------
        text : str
            Text to chunk
        
        Returns
        -------
        List[str]
            List of text chunks
        """
        return self._splitter.split_text(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        avg_size = 0
        if self._stats["chunks_created"] > 0:
            # Calculate from recent chunks
            avg_size = self.config.chunk_size  # Approximate
        
        return {
            **self._stats,
            "avg_chunk_size": avg_size,
            "config": {
                "strategy": self.config.strategy.value,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        }


# ==================== Convenience Functions ====================

def create_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive"
) -> List[Document]:
    """
    Convenience function to chunk documents.
    
    Parameters
    ----------
    documents : List[Document]
        Documents to chunk
    chunk_size : int
        Target chunk size
    chunk_overlap : int
        Overlap between chunks
    strategy : str
        Chunking strategy
    
    Returns
    -------
    List[Document]
        Chunked documents
    """
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy(strategy)
    )
    
    return chunker.chunk_documents(documents)


def analyze_chunks(chunks: List[Document]) -> Dict[str, Any]:
    """
    Analyze chunk distribution.
    
    Returns statistics about chunk sizes and distribution.
    """
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_characters": sum(sizes),
        "avg_chunk_size": sum(sizes) / len(sizes) if sizes else 0,
        "min_chunk_size": min(sizes) if sizes else 0,
        "max_chunk_size": max(sizes) if sizes else 0,
        "sources": list(set(
            chunk.metadata.get("source", "unknown") 
            for chunk in chunks
        ))
    }