---
id: project_career_copilot_agent
title: "Project: Career Copilot Agent"
tags: [project, langgraph, rag, llm, mcp, fastapi, react, docker, evals, ci]
repo: momo840505/career-copilot-agent
live_demo: https://career-copilot-agent.onrender.com
---

# Career Copilot Agent

Built an evidence-grounded job application assistant that parses job descriptions,
retrieves relevant resume and portfolio evidence, classifies skill gaps, and drafts
cover letters with source citations.

The workflow uses LangGraph for orchestration, structured Pydantic outputs, bounded
repair retries, a writer/critic revision loop, and a human approval checkpoint before
a draft is finalized. The same pipeline is exposed through FastAPI and an MCP server.

The retrieval layer uses ChromaDB with OpenAI embeddings and MMR re-ranking. Evaluation
includes deterministic citation and gap checks plus a golden job-description suite.
GitHub Actions runs unit tests, live evaluation checks, and a Docker boot smoke test.

## Stack
Python, LangGraph, LangChain, OpenAI API, ChromaDB, FastAPI, MCP, React, Vite, SQLite,
Docker, GitHub Actions, Render.

## What this demonstrates
Agent workflow design, grounded retrieval, LLM output validation, human-in-the-loop
control, evaluation, API design, deployment, and operational monitoring.
