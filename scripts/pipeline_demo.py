"""CLI: run the full parse_jd -> retrieve_evidence -> gap_analysis chain by hand.

This wires the three nodes together manually (plain function calls) so it's obvious
what each one does before Phase 4 turns this into an actual LangGraph state machine
with a self-correction loop and a human-approval checkpoint.

    python scripts/pipeline_demo.py path/to/jd.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_copilot.graph.gap_analysis import gap_analysis
from career_copilot.graph.parse_jd import parse_jd
from career_copilot.graph.retrieve_evidence import retrieve_evidence


def main() -> None:
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        jd_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        print(
            "Paste the job description, then press Ctrl-Z then Enter (Windows) "
            "or Ctrl-D (mac/linux):"
        )
        jd_text = sys.stdin.read()

    print("\n=== 1. parse_jd ===")
    jd = parse_jd(jd_text)
    print(jd.model_dump_json(indent=2))

    print("\n=== 2. retrieve_evidence ===")
    bundles = retrieve_evidence(jd)
    for b in bundles:
        chunk_ids = [c.chunk_id for c in b.chunks] or ["(none found)"]
        print(f"  {b.requirement}: {chunk_ids}")

    print("\n=== 3. gap_analysis ===")
    report = gap_analysis(jd, bundles)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
