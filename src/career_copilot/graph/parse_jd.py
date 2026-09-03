"""Phase 2, node 1: turn raw job-description text into validated JDRequirements."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from career_copilot.config import Settings, get_settings
from career_copilot.graph.structured import invoke_structured
from career_copilot.llm import get_chat_model
from career_copilot.schemas.jd import JDRequirements

_SYSTEM_PROMPT = """You are an expert technical recruiter. Extract structured requirements \
from a job description.

Rules:
- Only extract what the JD actually says. Do not invent skills, tools, or requirements that \
are not stated or clearly implied by the text.
- Distinguish clearly between "must have" (required — "must", "required", listed under a \
"Requirements" heading) and "nice to have" (bonus — "plus", "preferred", "nice to have").
- `keywords` should be the specific tools/technologies/domain terms a recruiter or ATS would \
search for (e.g. "Python", "AWS", "stakeholder management"), not generic filler words like \
"team player" or "fast-paced environment".
- If a single bullet lists interchangeable alternatives (e.g. "SQL or Python", "系統測試或自動化\
測試經驗" — either one satisfies it), extract it as ONE requirement, worded to keep both options \
(e.g. "SQL or Python"). Do NOT split it into two separate near-duplicate requirements — that loses \
the fact that either one was acceptable, and hands the rest of the pipeline two confusingly \
similar entries to reason about instead of the one real requirement the JD actually stated.
"""


def parse_jd(jd_text: str, settings: Settings | None = None) -> JDRequirements:
    settings = settings or get_settings()
    llm = get_chat_model(settings)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Job description:\n\n{jd_text.strip()}"),
    ]
    return invoke_structured(llm, JDRequirements, messages, node_name="parse_jd")
