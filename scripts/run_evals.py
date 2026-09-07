"""CLI: run the golden JD set (src/career_copilot/data/golden_jds/*.txt) through the
full pipeline and report objective + groundedness metrics for each. Exits non-zero if
any HARD gate fails (citation validity, no missing-skill
leak, critic convergence); the groundedness judge score is reported but only warns,
since it's noisier by nature (see groundedness_judge.py) and isn't held to a hard
threshold here.

    python scripts/run_evals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_copilot.config import get_settings
from career_copilot.eval.groundedness_judge import judge_groundedness
from career_copilot.eval.metrics import body_claim_coverage, citation_validity, critic_converged, no_missing_skill_leak
from career_copilot.graph.pipeline import MAX_REVISIONS, run_pipeline
from career_copilot.graph.structured import StructuredOutputError

GROUNDEDNESS_WARN_THRESHOLD = 0.7
GROUNDEDNESS_SPREAD_WARN = 0.4


def main() -> int:
    settings = get_settings()
    jd_files = sorted(settings.golden_jds_dir.glob("*.txt"))
    if not jd_files:
        print(f"No golden JD files found under {settings.golden_jds_dir}")
        return 1

    overall_ok = True
    for jd_file in jd_files:
        print(f"\n{'=' * 60}\n{jd_file.name}\n{'=' * 60}")
        jd_text = jd_file.read_text(encoding="utf-8")
        try:
            result = run_pipeline(jd_text, settings=settings)
        except StructuredOutputError as e:
            print(f"  [FAIL] pipeline could not produce valid output: {e}")
            overall_ok = False
            continue

        hard_checks = [
            citation_validity(result.draft, result.evidence_bundles),
            body_claim_coverage(result.draft),
            no_missing_skill_leak(result.draft, result.gap_report),
            critic_converged(result.revision_count, result.critic_verdict, MAX_REVISIONS),
        ]
        for check in hard_checks:
            status = "PASS" if check.passed else "FAIL"
            suffix = f" — {check.detail}" if check.detail else ""
            print(f"  [{status}] {check.name}{suffix}")
            if not check.passed:
                overall_ok = False

        groundedness = judge_groundedness(result.draft, result.evidence_bundles, settings=settings)
        status = "PASS" if groundedness.median_score >= GROUNDEDNESS_WARN_THRESHOLD else "WARN"
        print(
            f"  [{status}] groundedness (informational, not a hard gate): "
            f"median={groundedness.median_score:.2f} votes={groundedness.votes} "
            f"spread={groundedness.spread:.2f}"
        )
        if groundedness.spread > GROUNDEDNESS_SPREAD_WARN:
            print("         high spread across votes — judge disagreement, treat this score with caution")

    print(f"\n{'=' * 60}")
    print("ALL HARD CHECKS PASSED" if overall_ok else "SOME HARD CHECKS FAILED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
