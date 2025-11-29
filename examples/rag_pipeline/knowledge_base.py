"""Simple in-memory knowledge base for RAG example.

This module provides a minimal vector store implementation using numpy for
demonstrating RAG patterns without requiring heavy dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with content, metadata, and embedding.
    
    Attributes:
        content: The text content of the document.
        metadata: Additional metadata about the document.
        embedding: The vector embedding for similarity search.
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class SimpleKnowledgeBase:
    """A simple in-memory vector store using numpy cosine similarity.
    
    This implementation uses deterministic random embeddings based on text hashes
    for demonstration purposes. In production, use real embedding models.
    
    Args:
        embedding_fn: Optional custom embedding function. If None, uses fake embeddings.
        embedding_dim: Dimension of embedding vectors (default: 64).
    
    Example:
        >>> kb = SimpleKnowledgeBase()
        >>> kb.add_documents(["Hello world", "How are you?"])
        >>> results = kb.search("greeting", k=1)
        >>> print(results[0].content)
        Hello world
    """
    
    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        embedding_dim: int = 64,
    ):
        self.documents: list[Document] = []
        self.embedding_fn = embedding_fn
        self.embedding_dim = embedding_dim
        
    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add documents to the knowledge base.
        
        Args:
            texts: List of document texts to add.
            metadatas: Optional list of metadata dicts (one per document).
        
        Raises:
            ValueError: If texts and metadatas lists have different lengths.
        
        Note:
            This uses fake embeddings by default. For production use:
            ```python
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            kb = SimpleKnowledgeBase(embedding_fn=model.encode)
            ```
        """
        if not texts:
            logger.warning("add_documents called with empty texts list")
            return
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        if len(texts) != len(metadatas):
            raise ValueError(
                f"texts and metadatas must have same length: "
                f"{len(texts)} != {len(metadatas)}"
            )
            
        for text, metadata in zip(texts, metadatas):
            try:
                embedding = self._get_embedding(text)
                self.documents.append(Document(
                    content=text,
                    metadata=metadata,
                    embedding=embedding
                ))
            except Exception as e:
                logger.error(f"Failed to add document '{text[:50]}...': {e}")
                # Continue with other documents
                continue
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text.
        
        Args:
            text: The text to embed.
        
        Returns:
            Normalized embedding vector.
        """
        if self.embedding_fn is not None:
            # Use custom embedding function
            embedding = self.embedding_fn(text)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
        else:
            # Use fake embeddings (deterministic based on text hash)
            rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
            embedding = rng.random(self.embedding_dim)
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            logger.warning(f"Zero-norm embedding for text: {text[:50]}")
            
        return embedding
            
    def search(self, query: str, k: int = 3) -> list[Document]:
        """Search for most similar documents using cosine similarity.
        
        Args:
            query: The search query text.
            k: Number of documents to return (default: 3).
        
        Returns:
            List of up to k most similar documents, sorted by similarity score.
            Returns empty list if knowledge base is empty.
        
        Example:
            >>> kb = SimpleKnowledgeBase()
            >>> kb.add_documents(["Python is great", "I love coding"])
            >>> results = kb.search("programming", k=1)
        """
        if not self.documents:
            logger.warning("Search called on empty knowledge base")
            return []
        
        if k <= 0:
            logger.warning(f"Invalid k={k}, using k=1")
            k = 1
            
        try:
            # Get query embedding  
            query_embedding = self._get_embedding(query)
            
            # Calculate cosine similarity with all documents
            scores: list[tuple[float, Document]] = []
            for doc in self.documents:
                if doc.embedding is not None:
                    score = float(np.dot(query_embedding, doc.embedding))
                    scores.append((score, doc))
                else:
                    logger.warning(f"Document '{doc.content[:50]}' has no embedding")
            
            # Sort by score descending
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Return top k documents
            return [doc for _, doc in scores[:k]]
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []


__all__ = ["SimpleKnowledgeBase", "Document"]
