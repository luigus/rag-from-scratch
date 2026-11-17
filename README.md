# rag-from-scratch

A RAG (Retrieval-Augmented Generation) service built in phases. Each folder is a complete, runnable snapshot of the project, so you can compare one phase with the next and see exactly what changed and why.

| Phase | Folder | Topic |
|---|---|---|
| 1 | [`phase-01-basic-rag`](phase-01-basic-rag) | Upload, chunking, embeddings, k-NN search in OpenSearch, answers with sources |
| 2 | [`phase-02-evaluation`](phase-02-evaluation) | Golden dataset, retrieval metrics (recall@k, MRR), faithfulness with LLM-as-judge |
| 3 | `phase-03-hybrid-retrieval` | BM25 + vectors, Reciprocal Rank Fusion, reranking, measured against phase 2 |
| 4 | `phase-04-code-quality` | ruff, mypy, pre-commit, CI with an evaluation quality gate |
| 5 | `phase-05-production` | Structured outputs, retries and timeouts, caching, cost and latency logs |
| 6 | `phase-06-graphrag` | Entities and relations in a graph, used during retrieval |
| 7 | `phase-07-mcp` | The search exposed as a Model Context Protocol server |

Stack: Python 3.11+, FastAPI, Pydantic, Gemini (or OpenAI), OpenSearch, Docker.

To see what a phase changed: `diff -ru phase-01-basic-rag phase-02-evaluation`
