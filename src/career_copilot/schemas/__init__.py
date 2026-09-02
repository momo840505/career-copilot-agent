"""Pydantic schemas for every structured LLM output in the graph.

Treat the model's raw JSON as untrusted input — same as a user form submission — and
validate it for real (non-empty lists, no self-contradiction), not just "did it parse".
"""
