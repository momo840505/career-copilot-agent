"""CLI: paste a JD and see the structured extraction.

    python scripts/parse_jd_demo.py path/to/jd.txt
    # or with no file arg: paste text, then Ctrl-Z then Enter (Windows) / Ctrl-D (mac/linux)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_copilot.graph.parse_jd import parse_jd


def main() -> None:
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        jd_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        print(
            "Paste the job description, then press Ctrl-Z then Enter (Windows) "
            "or Ctrl-D (mac/linux):"
        )
        jd_text = sys.stdin.read()

    result = parse_jd(jd_text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
