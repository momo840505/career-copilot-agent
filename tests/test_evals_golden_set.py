"""The actual eval-suite-as-a-test: runs every golden JD through the full pipeline and
asserts the hard gates from career_copilot.eval.metrics. This is what
.github/workflows/ci.yml's eval-gate job runs — real API calls, real cost, so it's
marked requires_api and skipped automatically without an OPENAI_API_KEY (see
pytest.ini / the skipif pattern already used by every other requires_api test)."""
import os

import pytest

from career_copilot.config import get_settings
from career_copilot.eval.metrics import (
    body_claim_coverage,
    citation_validity,
    critic_converged,
    no_missing_skill_leak,
)
from career_copilot.graph.pipeline import MAX_REVISIONS, run_pipeline

pytestmark = pytest.mark.requires_api

needs_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set — skipping live API test"
)


def _golden_jd_paths() -> list:
    settings = get_settings()
    return sorted(settings.golden_jds_dir.glob("*.txt"))


@needs_key
@pytest.mark.parametrize("jd_path", _golden_jd_paths(), ids=lambda p: p.stem)
def test_golden_jd_passes_hard_gates(jd_path):
    jd_text = jd_path.read_text(encoding="utf-8")
    result = run_pipeline(jd_text)

    checks = [
        citation_validity(result.draft, result.evidence_bundles),
        body_claim_coverage(result.draft),
        no_missing_skill_leak(result.draft, result.gap_report),
        critic_converged(result.revision_count, result.critic_verdict, MAX_REVISIONS),
    ]
    failures = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    assert not failures, "\n".join(failures)
