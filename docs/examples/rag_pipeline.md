# RAG Pipeline Example

This example (`examples/rag_pipeline`) demonstrates how to evaluate Retrieval-Augmented Generation systems.

## Key Components
- **Knowledge Base**: Simulates a document store.
- **Retriever**: Fetches relevant documents based on the query.
- **Generator**: Synthesizes the answer using retrieved context.

## Custom Runner
This example uses a custom `RetrievalAugmentedRunner` to intercept the prompt and inject context before sending it to the model.

## Running the Example

```bash
uv run python -m examples.rag_pipeline.cli run
```
