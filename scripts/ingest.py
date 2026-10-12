"""Validate data/corpus.json and export each document as its own JSON file for inspection."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.corpus import load_corpus  # noqa: E402

docs = load_corpus()
target = ROOT / "data" / "documents"
target.mkdir(parents=True, exist_ok=True)
for doc in docs:
    (target / f"{doc.document_id}.json").write_text(json.dumps(doc.to_dict(), indent=2))
families = {d.family for d in docs}
print(f"Validated {len(docs)} documents in {len(families)} policy families; exported to {target}")
