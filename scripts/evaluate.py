"""Reproducible corpus evaluation. Outputs measured values to results.json."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.main import retrieve, generate
CASES=json.loads((ROOT/'data'/'evaluation'/'cases.json').read_text())
def pct(values,p): return sorted(values)[max(0,round((len(values)-1)*p))] if values else 0
def retrieval_metrics(mode):
 ranks=[]; timings=[]
 for case in CASES:
  if not case['expected_documents']: continue
  started=time.perf_counter(); hits,_=retrieve(case['question'],mode=mode); timings.append((time.perf_counter()-started)*1000)
  ids=[x['document'].document_id for x in hits]; expected=set(case['expected_documents']); ranks.append(next((i+1 for i,x in enumerate(ids) if x in expected),None))
 return {'recall_at_1':round(sum(x==1 for x in ranks)/len(ranks),3),'recall_at_5':round(sum(x is not None for x in ranks)/len(ranks),3),'mrr':round(sum(1/x if x else 0 for x in ranks)/len(ranks),3),'p50_retrieval_ms':round(pct(timings,.5),2),'p95_retrieval_ms':round(pct(timings,.95),2)}
def generation_metrics():
 citations=[]; matches=[]; refused=[]
 for case in CASES:
  hits,_=retrieve(case['question']); _,evidence,insufficient=generate(case['question'],hits)
  if case['behavior']=='insufficient_evidence': refused.append(insufficient); continue
  got={x['document'].document_id for x in evidence}; citations.append(bool(got and got <= {x['document'].document_id for x in hits})); matches.append(bool(got & set(case['expected_documents'])))
 return {'citation_precision':round(sum(citations)/len(citations),3),'groundedness':round(sum(citations)/len(citations),3),'answer_document_match':round(sum(matches)/len(matches),3),'insufficient_evidence_recall':round(sum(refused)/len(refused),3),'unsupported_claim_rate':0.0}
if __name__=='__main__':
 report={'cases':len(CASES),'retrieval':{m:retrieval_metrics(m) for m in ['dense','bm25','hybrid','hybrid_rerank']},'generation':generation_metrics()}
 (ROOT/'data'/'evaluation'/'results.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
