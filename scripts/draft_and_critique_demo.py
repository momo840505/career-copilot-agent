"""CLI: run the full chain including the draft_writer <-> critic self-correction
loop, by hand (a plain Python loop) — before Phase 4b turns this into an actual
LangGraph StateGraph with conditional edges and a human-approval interrupt.

    python scripts/draft_and_critique_demo.py path/to/jd.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_copilot.graph.critic import critic
from career_copilot.graph.draft_writer import draft_writer
from career_copilot.graph.gap_analysis import gap_analysis
from career_copilot.graph.parse_jd import parse_jd
from career_copilot.graph.retrieve_evidence import retrieve_evidence
from career_copilot.graph.structured import StructuredOutputError

MAX_REVISIONS = 2


def main() -> None:
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        jd_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        print(
            "Paste the job description, then press Ctrl-Z then Enter (Windows) "
            "or Ctrl-D (mac/linux):"
        )
        jd_text = sys.stdin.read()

    jd = parse_jd(jd_text)
    bundles = retrieve_evidence(jd)
    try:
        report = gap_analysis(jd, bundles)
    except StructuredOutputError as e:
        print(f"\n*** gap_analysis could not produce a valid report: {e}\n")
        print("This means every repair attempt failed — worth re-running once (LLM output")
        print("is non-deterministic), and if it keeps happening, worth raising max_retries")
        print("or trying a stronger OPENAI_CHAT_MODEL.")
        return

    print("\n=== gap_analysis ===")
    print(report.model_dump_json(indent=2))

    feedback_history: list[str] = []  # accumulates across attempts — see draft_writer.py
    for attempt in range(1, MAX_REVISIONS + 2):  # 1 initial + MAX_REVISIONS repairs
        print(f"\n=== draft_writer (attempt {attempt}) ===")
        try:
            draft = draft_writer(jd, report, bundles, revision_feedback=feedback_history or None)
        except StructuredOutputError as e:
            print(f"\n*** draft_writer could not produce a valid draft: {e}\n")
            return
        print(draft.model_dump_json(indent=2))

        print(f"\n=== critic (attempt {attempt}) ===")
        try:
            verdict = critic(draft, report, bundles)
        except StructuredOutputError as e:
            print(f"\n*** critic could not produce a valid verdict: {e}\n")
            return
        print(verdict.model_dump_json(indent=2))

        if verdict.passed:
            print("\n*** Critic passed. Draft would go to human review next. ***")
            break

        if attempt == MAX_REVISIONS + 1:
            print("\n*** Max revisions reached and critic still not satisfied. Stopping. ***")
            break

        new_feedback = verdict.issues + [
            f'Claim "{c.claim_text}" is not well-grounded: {c.reason}' for c in verdict.ungrounded_claims
        ]
        for item in new_feedback:
            if item not in feedback_history:  # don't pile up exact duplicates
                feedback_history.append(item)
        print(f"\n--> Sending back to draft_writer with full feedback history: {feedback_history}")


if __name__ == "__main__":
    main()
