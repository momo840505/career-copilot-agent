"""FastAPI service exposing the pipeline over HTTP.

Kept as a thin package (just this file + app.py) mirroring the rest of the project's
"one small module per concern" style -- no reason for this to be a single flat module
once tests and possibly future routers live alongside it.
"""
