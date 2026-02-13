import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` can be imported by pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import agent as agent_mod


def test_wizard_lookup_schema():
    out = agent_mod._wizard_lookup("Dumbledore")
    parsed = json.loads(out)
    assert "query" in parsed
    assert isinstance(parsed["query"], str)
    assert "potions" in parsed
    assert isinstance(parsed["potions"], list)
    for item in parsed["potions"]:
        assert isinstance(item, dict)
        if "error" in item:
            # allow error objects
            continue
        assert "wizard" in item
        assert "potion_name" in item
        assert "potion_description" in item
