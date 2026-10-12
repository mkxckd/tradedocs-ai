"""SQLite-backed telemetry and feedback repository; no secrets are persisted."""
import json
import os
import sqlite3
from pathlib import Path
from threading import Lock

class TelemetryStore:
    def __init__(self, db_path: str | None = None):
        self.path = db_path or os.getenv("TRADEDOCS_DB") or str(Path(__file__).parents[2] / "data" / "runtime" / "tradedocs.db")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS query_events (
                request_id TEXT PRIMARY KEY, timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                query_text TEXT NOT NULL, intent TEXT NOT NULL, rewritten_query TEXT,
                retrieval_ms REAL, generation_ms REAL, total_ms REAL,
                retrieved_ids TEXT NOT NULL, citation_ids TEXT NOT NULL, insufficient INTEGER NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS feedback (
                request_id TEXT NOT NULL, helpful INTEGER NOT NULL, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)""")
    def _connect(self):
        return sqlite3.connect(self.path)
    def record(self, event: dict):
        with self.lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO query_events(request_id,query_text,intent,rewritten_query,retrieval_ms,generation_ms,total_ms,retrieved_ids,citation_ids,insufficient) VALUES(?,?,?,?,?,?,?,?,?,?)", (event['request_id'],event['query'],event['intent'],event['rewritten_query'],event['retrieval_ms'],event['generation_ms'],event['total_ms'],json.dumps(event['retrieved_document_ids']),json.dumps(event['citations']),int(event['insufficient_evidence'])))
    def feedback(self, request_id: str, helpful: bool):
        with self.lock, self._connect() as db:
            db.execute("INSERT INTO feedback(request_id,helpful) VALUES(?,?)",(request_id,int(helpful)))
    def latencies(self) -> list[float]:
        with self._connect() as db:
            return sorted(r[0] for r in db.execute("SELECT total_ms FROM query_events WHERE total_ms IS NOT NULL"))

    def metrics(self):
        with self._connect() as db:
            total=db.execute("SELECT count(*) FROM query_events").fetchone()[0]
            avg=db.execute("SELECT avg(total_ms) FROM query_events").fetchone()[0] or 0
            refused=db.execute("SELECT count(*) FROM query_events WHERE insufficient=1").fetchone()[0]
            votes=db.execute("SELECT count(*),avg(helpful) FROM feedback").fetchone()
            intents=dict(db.execute("SELECT intent,count(*) FROM query_events GROUP BY intent").fetchall())
            low=list(db.execute("SELECT q.query_text,q.request_id FROM query_events q JOIN feedback f ON q.request_id=f.request_id WHERE f.helpful=0 ORDER BY f.timestamp DESC LIMIT 10").fetchall())
        return {'total_queries':total,'helpful_rate':votes[1] if votes[0] else None,'feedback_count':votes[0],'unanswerable_rate':refused/total if total else 0,'average_latency_ms':round(avg,2),'intents':intents,'low_rated_questions':[{'query':q,'request_id':r} for q,r in low]}
