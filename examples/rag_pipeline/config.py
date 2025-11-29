"""Configuration for RAG example."""
from dataclasses import dataclass, field
from typing import List
from themis.config import ExperimentConfig as BaseConfig

@dataclass
class RagConfig(BaseConfig):
    """Extended configuration for RAG experiment."""
    # RAG specific settings
    knowledge_base_docs: List[str] = field(default_factory=list)
    retrieval_k: int = 2
