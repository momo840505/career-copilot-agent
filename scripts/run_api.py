"""CLI: launch the FastAPI service (Phase 6).

    python scripts/run_api.py

Docs at http://127.0.0.1:8000/docs once it's running.

Deliberately does NOT use the `sys.path.insert(...)` pattern the other scripts in this
folder use (see run_evals.py, graph_demo.py, etc.). Those scripts import
career_copilot directly in the same process, so patching sys.path before the import
works fine. This script instead hands uvicorn a STRING app target ("career_copilot.
api.app:app") with reload=True, and uvicorn's reloader runs the actual server in a
SEPARATE subprocess that re-imports that string fresh — a sys.path.insert() done here,
in the parent process, would simply not exist in that subprocess, and the import would
fail with ModuleNotFoundError the moment reload was enabled. The fix is to rely on the
editable install instead (`pip install -e .`, already part of this project's documented
setup and the CI workflow) so `career_copilot` is importable from any process without
any path patching at all.
"""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("career_copilot.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
