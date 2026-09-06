"""Career Copilot: a multi-agent job-application assistant.

Package layout:
    config.py       - environment/config loading
    schemas/        - Pydantic models for every structured LLM output
    data/portfolio/ - source-of-truth markdown docs (resume + project write-ups)
    rag/            - chunking, embedding, vector store, retrieval
    graph/          - LangGraph nodes + graph assembly
    api/            - FastAPI service + MCP server
"""
