"""Export the self-contained fictional corpus as inspectable JSON documents."""
import json,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.main import DOCS
target=ROOT/'data'/'documents'; target.mkdir(parents=True,exist_ok=True)
for doc in DOCS:
    (target/f'{doc.document_id}.json').write_text(json.dumps(asdict(doc),indent=2))
print(f'Exported {len(DOCS)} TradeFlow documents to {target}')
