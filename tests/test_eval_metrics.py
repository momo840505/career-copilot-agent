"""Pure-logic tests for career_copilot.eval.metrics — no API key needed. These lock in
the objective, code-based CI gates independent of any live pipeline run."""
from career_copilot.eval.metrics import citation_validity, critic_converged, no_missing_skill_leak
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.critic import CriticVerdict
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport


def _bundle(chunk_id: str, text: str = "some evidence text") -> EvidenceBundle:
    chunk = RetrievedChunk(chunk_id=chunk_id, doc_id="doc", doc_title="Doc", text=text, distance=0.1)
    return EvidenceBundle(requirement="req", chunks=[chunk])


def _draft(body: str, claims: list[Claim]) -> CoverLetterDraft:
    return CoverLetterDraft(greeting="Dear Hiring Manager,", body=body, closing="Best,", claims=claims)


def test_citation_validity_passes_with_real_chunk_ids():
    bundles = [_bundle("skills::chunk0")]
    draft = _draft("Body.", [Claim(text="A claim.", evidence_chunk_ids=["skills::chunk0"])])
    result = citation_validity(draft, bundles)
    assert result.passed


def test_citation_validity_fails_with_a_hallucinated_chunk_id():
    bundles = [_bundle("skills::chunk0")]
    draft = _draft("Body.", [Claim(text="A claim.", evidence_chunk_ids=["made_up::chunk9"])])
    result = citation_validity(draft, bundles)
    assert not result.passed
    assert "made_up::chunk9" in result.detail


def test_no_missing_skill_leak_passes_when_missing_item_absent_from_body():
    gap_report = GapReport(
        missing=[GapItem(requirement="Kubernetes", evidence_chunk_ids=[], note="no evidence")],
        overall_fit_summary="ok",
    )
    draft = _draft("I have strong Python and SQL experience.", [Claim(text="x", evidence_chunk_ids=["a"])])
    assert no_missing_skill_leak(draft, gap_report).passed


def test_no_missing_skill_leak_fails_on_undisclaimed_mention():
    gap_report = GapReport(
        missing=[GapItem(requirement="Kubernetes", evidence_chunk_ids=[], note="no evidence")],
        overall_fit_summary="ok",
    )
    draft = _draft(
        "I have deep hands-on Kubernetes expertise from years of production work.",
        [Claim(text="x", evidence_chunk_ids=["a"])],
    )
    result = no_missing_skill_leak(draft, gap_report)
    assert not result.passed
    assert "Kubernetes" in result.detail


def test_no_missing_skill_leak_passes_when_mention_is_disclaimed():
    gap_report = GapReport(
        missing=[GapItem(requirement="Kubernetes", evidence_chunk_ids=[], note="no evidence")],
        overall_fit_summary="ok",
    )
    draft = _draft(
        "While I don't have direct Kubernetes experience, I bring strong adjacent skills.",
        [Claim(text="x", evidence_chunk_ids=["a"])],
    )
    assert no_missing_skill_leak(draft, gap_report).passed


def test_critic_converged_true_when_passed():
    verdict = CriticVerdict(passed=True)
    result = critic_converged(revision_count=1, critic_verdict=verdict, max_revisions=2)
    assert result.passed


def test_critic_converged_false_when_not_passed():
    verdict = CriticVerdict(passed=False, issues=["still generic"])
    result = critic_converged(revision_count=2, critic_verdict=verdict, max_revisions=2)
    assert not result.passed
    assert "2/2" in result.detail
