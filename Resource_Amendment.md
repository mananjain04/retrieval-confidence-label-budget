# Resource amendment before calibration analysis

The original v2 plan used BM25, MiniLM and BGE-small. Both contextual ONNX full-corpus runs were interrupted before completion. Concurrent memory pressure also interrupted the first SciDocs sparse retrieval attempt; its clean rerun completed. No contextual model is counted as an executed baseline. A separate BGE-small CPU feasibility pilot encoded16 long documents in31.8857 seconds; this timing is not a retrieval-quality result.

The v3 protocol replaces contextual inference with two pinned static embeddings, POTION-base-8M and POTION-retrieval-32M. The replacement was frozen before the v3 embedding runs and before any v3 calibration metrics were computed. BM25 feature artifacts from the completed lexical runs were retained, with unknown interrupted-registry timing recorded as null. No calibration result was used to choose the replacement models.

The scope is therefore lexical and static dense retrieval on four collections. It does not support conclusions about contextual transformers, rerankers, production RAG, or generated answers. Earlier v1 results remain a separate, previously seen exploratory study; the extension is not a blinded discovery or a public preregistration.
