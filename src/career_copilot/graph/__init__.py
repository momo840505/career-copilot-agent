"""LangGraph nodes, plus the assembled graph.

parse_jd.py, retrieve_evidence.py, and gap_analysis.py extract and analyze the JD.
draft_writer.py, critic.py, and human_review.py handle the drafting loop, with
build_graph.py wiring it all together with conditional edges (the self-correction
loop).
"""
