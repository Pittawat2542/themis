"""RAG Generation Runner implementation.

This module implements a custom generation runner that performs retrieval-augmented
generation by searching a knowledge base before each generation request.
"""
from __future__ import annotations

import logging
from typing import Optional

from themis.core import entities as core_entities
from themis.generation.runner import GenerationRunner

from .knowledge_base import SimpleKnowledgeBase

logger = logging.getLogger(__name__)


class RetrievalAugmentedRunner(GenerationRunner):
    """A custom runner that performs retrieval before generation.
    
    This runner intercepts generation tasks, searches a knowledge base for relevant
    context, and augments the prompt with retrieved information before calling the
    underlying provider.
    
    Args:
        provider: The LLM provider to use for generation.
        knowledge_base: The knowledge base to search for relevant documents.
        k: Number of documents to retrieve (default: 2).
        context_template: Optional custom template for formatting context.
        **kwargs: Additional arguments passed to GenerationRunner.
    
    Example:
        >>> kb = SimpleKnowledgeBase()
        >>> kb.add_documents(["Python is a programming language"])
        >>> runner = RetrievalAugmentedRunner(
        ...     provider=my_provider,
        ...     knowledge_base=kb,
        ...     k=3
        ... )
    """
    
    def __init__(
        self,
        *,
        provider,
        knowledge_base: SimpleKnowledgeBase,
        k: int = 2,
        context_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(provider=provider, **kwargs)
        self.knowledge_base = knowledge_base
        self.k = max(1, k)  # Ensure k is at least 1
        self.context_template = context_template or self._default_template()
        
    @staticmethod
    def _default_template() -> str:
        """Get the default context template."""
        return (
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query}\n"
            "Answer:"
        )
        
    def _extract_prompt_text(self, task: core_entities.GenerationTask) -> str:
        """Extract text from task prompt.
        
        Args:
            task: The generation task.
        
        Returns:
            The prompt text as a string.
        """
        if hasattr(task.prompt, 'text'):
            return task.prompt.text
        elif hasattr(task.prompt, 'prompt_text'):
            return task.prompt.prompt_text
        else:
            return str(task.prompt)
        
    def _generate_single(
        self,
        task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        """Generate a single response with retrieval augmentation.
        
        Args:
            task: The generation task containing the query.
        
        Returns:
            A generation record with the response.
        
        Note:
            This method:
            1. Extracts the query from the task
            2. Retrieves relevant documents from the knowledge base
            3. Augments the prompt with retrieved context
            4. Calls the provider with the augmented prompt
            5. Returns a record linked to the original task
        """
        # 1. Extract query text
        query = self._extract_prompt_text(task)
        logger.debug(f"RAG query: {query[:100]}...")
        
        # 2. Retrieve relevant documents
        try:
            docs = self.knowledge_base.search(query, k=self.k)
            logger.debug(f"Retrieved {len(docs)} documents")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}. Continuing without context.")
            docs = []
        
        # 3. Format retrieved context
        if docs:
            context_str = "\n\n".join([
                f"Document {i+1}:\n{doc.content}" 
                for i, doc in enumerate(docs)
            ])
        else:
            context_str = "No relevant context found."
            logger.warning("No documents retrieved, using empty context")
        
        # 4. Augment prompt using template
        augmented_prompt = self.context_template.format(
            context=context_str,
            query=query
        )
        
        # 5. Create new task with augmented prompt
        augmented_prompt_spec = core_entities.PromptSpec(
            name="rag_augmented",
            template="{augmented_content}"
        )
        augmented_prompt_render = core_entities.PromptRender(
            spec=augmented_prompt_spec,
            text=augmented_prompt,
            context={"augmented_content": augmented_prompt}
        )
        
        augmented_task = core_entities.GenerationTask(
            prompt=augmented_prompt_render,
            model=task.model,
            sampling=task.sampling,
            metadata={
                **task.metadata,
                "retrieved_docs": [d.content for d in docs],
                "retrieved_doc_count": len(docs),
                "original_prompt": query
            }
        )
        
        # 6. Delegate to provider
        try:
            record = self._invoke_provider(augmented_task)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
        # 7. Return record linked to original task for evaluation consistency
        return core_entities.GenerationRecord(
            task=task,
            output=record.output,
            error=record.error,
            metrics={
                **record.metrics,
                "rag_docs_retrieved": len(docs)
            },
            attempts=record.attempts
        )


__all__ = ["RetrievalAugmentedRunner"]
