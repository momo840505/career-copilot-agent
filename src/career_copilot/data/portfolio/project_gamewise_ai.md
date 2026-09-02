---
id: project_gamewise_ai
title: "Project: GameWise AI — Steam Game Recommender"
tags: [project, nlp, semantic-search, sentence-transformers, streamlit, deployment]
repo: momo840505/gamewise-ai
live_demo: https://gamewise-ai.streamlit.app/
---

# GameWise AI — Steam Game Recommendation System

Takes a natural-language query (e.g. "two-player co-op survival game, budget under $20") and
automatically decomposes it into multiple filter conditions, combined with semantic search
(Sentence Transformers) so results capture relevance beyond literal keyword matching.

Passed all 11 hand-designed evaluation scenarios and is covered by 16 automated tests. If a
request is too vague, the system proactively asks a follow-up question rather than guessing;
if no results match, it says so honestly instead of recommending something irrelevant.
Deployed live, with support for favoriting, sorting, and CSV export.

## Stack
Python, Streamlit, Sentence Transformers, NLP.

## What this demonstrates
Semantic/embedding-based retrieval, scenario-based evaluation of an NLP system (11/11 passing
scenarios + 16 automated tests — an early version of the "eval-driven" mindset Career Copilot
formalizes further), and a UX principle carried forward into Career Copilot: don't fabricate
an answer when the system isn't confident — ask, or say so honestly.
