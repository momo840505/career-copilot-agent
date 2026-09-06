"""CLI: run the full career-copilot-agent pipeline as an actual LangGraph
StateGraph -- same pipeline as draft_and_critique_demo.py, but the
draft_writer <-> critic loop is now a real conditional edge instead of a
Python for-loop, and there's a genuine human-in-the-loop pause (via
interrupt()/Command(resume=...)) before a letter counts as final.

    python scripts/graph_demo.py path/to/jd.txt
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.types import Command

from career_copilot.graph.build_graph import build_graph
from career_copilot.graph.structured import StructuredOutputError


def _print_interrupt_payload(payload: dict) -> None:
    draft = payload["draft"]
    print("\n=== HUMAN REVIEW NEEDED ===")
    print(f"Greeting: {draft['greeting']}")
    print(f"\nBody:\n{draft['body']}")
    print(f"\nClosing: {draft['closing']}")
    print(f"\nCritic passed: {payload['critic_passed']}")
    if payload["critic_issues"]:
        print(f"Remaining issues critic still had: {payload['critic_issues']}")
    print(f"\nGap summary: {payload['gap_summary']}")


def _ask_human_decision() -> dict:
    while True:
        choice = input("\nApprove this draft? [y]es / [n]o (request a revision): ").strip().lower()
        if choice in ("y", "yes"):
            return {"action": "approve"}
        if choice in ("n", "no"):
            feedback = input("What should change? ").strip()
            return {"action": "revise", "feedback": feedback or "Please revise and try again."}
        print("Please answer y or n.")


def main() -> None:
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        jd_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        print(
            "Paste the job description, then press Ctrl-Z then Enter (Windows) "
            "or Ctrl-D (mac/linux):"
        )
        jd_text = sys.stdin.read()

    graph = build_graph()
    # A fresh thread_id per run — it's the key the checkpointer uses to find its way
    # back to THIS paused run when we later call Command(resume=...).
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        result = graph.invoke({"jd_text": jd_text}, config=config)
    except StructuredOutputError as e:
        print(f"\n*** Pipeline could not produce valid output: {e}\n")
        return

    # A node hit interrupt() -> invoke() returned normally (no exception) with a
    # "__interrupt__" key instead of running to completion. Keep resuming with the
    # human's decision until there's no interrupt left in the result.
    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        _print_interrupt_payload(interrupt_obj.value)
        decision = _ask_human_decision()
        try:
            result = graph.invoke(Command(resume=decision), config=config)
        except StructuredOutputError as e:
            print(f"\n*** Pipeline could not produce valid output during revision: {e}\n")
            return

    if result.get("human_decision", {}).get("action") == "approve":
        draft = result["draft"]
        print("\n*** Approved. Final cover letter: ***\n")
        print(f"{draft.greeting}\n\n{draft.body}\n\n{draft.closing}")
    else:
        print("\n*** Ended without approval. ***")


if __name__ == "__main__":
    main()
