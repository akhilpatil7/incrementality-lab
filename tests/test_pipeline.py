"""Program-level checks: summary.json is built from module results and the README scoreboard matches it."""
import json
from pathlib import Path

import pytest

from incrlab.config import RESULTS_DIR
from incrlab.pipeline import build_summary, headlines_markdown, scoreboard_markdown

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not all((RESULTS_DIR / f"{m}.json").exists() for m in ("meta", "search", "dsp", "mmm")),
                                reason="module results not generated")


def test_summary_has_every_module_and_truth():
    s = build_summary()
    assert set(s["modules"]) == {"meta", "search", "dsp", "mmm"}
    assert all(r["truth"] is not None for r in s["scoreboard"])


def test_readme_scoreboard_matches_summary_json():
    p = RESULTS_DIR / "summary.json"
    if not p.exists():
        pytest.skip("summary.json not generated")
    readme = (ROOT / "README.md").read_text()
    a, b = "<!-- SCOREBOARD:START -->", "<!-- SCOREBOARD:END -->"
    block = readme[readme.index(a) + len(a):readme.index(b)].strip()
    assert block == scoreboard_markdown(json.loads(p.read_text())).strip()
    a, b = "<!-- HEADLINES:START -->", "<!-- HEADLINES:END -->"
    block = readme[readme.index(a) + len(a):readme.index(b)].strip()
    assert block == headlines_markdown(json.loads(p.read_text())).strip()
