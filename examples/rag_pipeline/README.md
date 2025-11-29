# RAG Pipeline Example

This example demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using Themis.

## Prerequisites

- Python 3.9+
- Themis installed
- `numpy` (for vector operations)

## Features Demonstrated

- **Custom Generation Runner**: Intercepting the generation process to perform retrieval
- **In-memory Vector Store**: Simple knowledge base using numpy cosine similarity
- **Prompt Augmentation**: Injecting retrieved context into prompts
- **Task Metadata Tracking**: Recording retrieval results for analysis
- **Error Handling**: Graceful degradation when retrieval fails

## How it Works

### Architecture

```
User Query
    ↓
RetrievalAugmentedRunner
    ↓
SimpleKnowledgeBase.search()
    ↓
Retrieve top-k documents
   ↓
Augment prompt with context
    ↓
Send to Provider
    ↓
Return Response
```

### Components

1. **Knowledge Base** (`knowledge_base.py`): Simple in-memory vector store using numpy
   - Deterministic fake embeddings for demo (hash-based)
   - Support for custom embedding functions
   - Cosine similarity search

2. **Retriever** (`retriever.py`): Custom runner that performs RAG
   - Searches knowledge base for relevant context
   - Augments prompts with retrieved documents
   - Preserves original task for evaluation

3. **Experiment** (`experiment.py`): Wires everything together
   - Initializes knowledge base with sample documents
   - Creates custom runner with retrieval
   - Runs experiment and reports results

## Running the Example

```bash
# Run with default settings
uv run python -m examples.rag_pipeline.cli run

# Limit to specific number of samples
uv run python -m examples.rag_pipeline.cli run --limit 5
```

## Expected Output

```
Initializing knowledge base with 5 documents...
Running RAG experiment with 2 samples...

Experiment completed!

--- Example Generation ---
Augmented Prompt:
Context information is below.
---------------------
Document 1:
Themis is a lightweight experimentation harness for text-generation systems.
---------------------
Given the context information, answer the query.
Query: What is Themis?
Answer:
Response: {"answer": "...", "reasoning": "..."}
```

## Key Code Examples

### Creating a Knowledge Base

```python
from examples.rag_pipeline.knowledge_base import SimpleKnowledgeBase

# Simple fake embeddings (demo only)
kb = SimpleKnowledgeBase()
kb.add_documents([
    "Python is a programming language",
    "Themis evaluates LLM outputs"
])

# With real embeddings (production)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
kb = SimpleKnowledgeBase(embedding_fn=model.encode)
```

### Custom RAG Runner

```python
from examples.rag_pipeline.retriever import RetrievalAugmentedRunner
from themis.generation.clients import FakeMathModelClient

provider = FakeMathModelClient()
runner = RetrievalAugmentedRunner(
    provider=provider,
    knowledge_base=kb,
    k=3,  # retrieve top 3 documents
    context_template="Context: {context}\nQ: {query}\nA:"
)
```

## Extending This Example

### Use Real Embeddings

Install a sentence transformer model:

```bash
pip install sentence-transformers
```

Then modify the knowledge base initialization:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
kb = SimpleKnowledgeBase(embedding_fn=model.encode)
```

### Connect to Vector Database

Replace `SimpleKnowledgeBase` with a production vector DB like Chroma or Pinecone:

```python
import chromadb
from examples.rag_pipeline.knowledge_base import Document

class ChromaKnowledgeBase:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("docs")
    
    def add_documents(self, texts, metadatas=None):
        self.collection.add(
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            ids=[f"doc{i}" for i in range(len(texts))]
        )
    
    def search(self, query, k=3):
        results = self.collection.query(query_texts=[query], n_results=k)
        return [
            Document(content=doc, metadata=meta)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
```

### Custom Context Template

```python
runner = RetrievalAugmentedRunner(
    provider=provider,
    knowledge_base=kb,
    context_template=(
        "# Reference Material\n"
        "{context}\n\n"
        "# Question\n"
        "{query}\n\n"
        "# Answer\n"
    )
)
```

## Troubleshooting

**Issue**: No documents retrieved
- Check that knowledge base has documents: `len(kb.documents)`
- Verify search is being called: enable debug logging

**Issue**: Poor retrieval quality with fake embeddings
- Expected! Fake embeddings are random. Use real embeddings for production.

**Issue**: Memory usage too high
- Reduce embedding dimensions: `SimpleKnowledgeBase(embedding_dim=32)`
- Use a vector database instead of in-memory storage

## Related Examples

- **[advanced](../advanced/)**: More custom runner patterns
- **[litellm_example](../litellm_example/)**: Using real LLM providers

## Further Reading

- Themis [README](../../README.md) for core concepts
- RAG patterns and best practices
