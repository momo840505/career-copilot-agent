"""Career Copilot: a multi-agent job-application assistant.

Package layout:
    config.py       - environment/config loading
    schemas/        - Pydantic models for every structured LLM output (Phase 2+)
    data/portfolio/ - source-of-truth markdown docs (resume + project write-ups)
    rag/            - chunking, embedding, vector store, retrieval (Phase 1)
    graph/          - LangGraph nodes + graph assembly (Phase 2-4)
    api/            - FastAPI service + MCP server (Phase 6)
"""
