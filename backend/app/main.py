"""TradeDocs AI: inspectable, dependency-light RAG portfolio implementation."""
import json, math, os, re, time, uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .storage import TelemetryStore
from .generation import OpenAICompatibleGenerator

@dataclass(frozen=True)
class Document:
    document_id:str; title:str; category:str; version:str; date:str; author:str; tags:list; source_type:str; content:str

SEED=[
('api-rate-v32','API Rate Limits v3.2','api','3.2',['429','rate limit'],'The current REST Orders endpoint allowance is 180 requests per minute per account. A HTTP 429 response includes X-RateLimit-Remaining and Retry-After. Use exponential backoff with jitter.'),
('api-rate-v21','API Rate Limits v2.1','api','2.1',['429','legacy'],'Version 2.1 allowed 120 requests per minute for the REST Orders endpoint. This policy was superseded by version 3.2.'),
('api-errors','API Error Handling Guide v3.2','api','3.2',['429','retry'],'HTTP 429 means request throttling, not authentication failure. Respect Retry-After. HTTP 401 means the token is missing, expired, or invalid.'),
('api-auth','API Authentication v3.2','api','3.2',['oauth','api key'],'Use OAuth 2.0 client credentials for server integrations. Send the access token as Authorization: Bearer. Rotate exposed API keys.'),
('orders-rejected','Order Rejection Troubleshooting v1.4','trading','1.4',['rejected','order'],'Check instrument status, market hours, buying power, price bands, account permissions, and minimum order quantity. The rejection code identifies the failed validation.'),
('withdrawals','Withdrawal Requirements v2.0','account','2.0',['withdrawal','verification'],'Withdrawals require a verified account, two-factor authentication, and a linked payment method in the account holder name. A security hold can apply after a password reset or new bank account.'),
('latency-incident','Incident 2025-04: Elevated Order API Latency','operations','1.0',['latency','incident'],'Symptoms included elevated p95 order API latency and timeouts. Root cause was database connection pool exhaustion. Resolution increased pool capacity and added saturation alerts.'),
('throttle-incident','Incident 2025-09: REST API Throttling','operations','1.0',['429','incident'],'A traffic burst caused elevated 429 responses for 18 minutes. Existing endpoint limits remained in effect; customers were advised to honor Retry-After and backoff.'),
('websocket','WebSocket Connection Behavior v3.2','api','3.2',['websocket','reconnect'],'WebSocket connections send a heartbeat every 30 seconds. Reconnect with exponential backoff after an unexpected close and resubscribe.'),
('security-login','Suspicious Login Response v1.2','security','1.2',['login','takeover'],'Reset the password, revoke active sessions, rotate API keys, and contact support for an unfamiliar login. Never share one-time codes.'),
('pagination','Pagination v3.2','api','3.2',['cursor','pagination'],'List endpoints use cursor pagination. Pass next_cursor from the previous response. Do not assume stable offsets while orders are changing.'),
('market-hours','Market Hours v1.1','trading','1.1',['hours','market'],'Trading availability follows the market schedule displayed for each instrument. Holiday schedules and volatility halts can change availability.'),
]
TOPICS=['retries','cancellation','verification','contract types','service degradation','account restrictions','authentication failures','debugging','execution rules','payment holds','order submission','API endpoints','trading permissions','incident response','monitoring','error codes','risk controls','websocket recovery','request tracing','support workflow','market data','client timeouts','status codes','account recovery','release notes','audit events','service health','idempotency','data retention','notifications','maintenance','settlement','order types','deposits','limits','escalation','key rotation','outages','logging','observability']
DOCS=[Document(a,b,c,d,'2025-10-01' if d=='3.2' else '2025-04-10','TradeFlow Docs',e,'incident' if 'Incident' in b else 'documentation',f) for a,b,c,d,e,f in SEED]
DOCS += [Document(f'guide-{i:02}',f'{t.title()} Guide','operations' if i%5==0 else 'api','1.0','2025-06-15','TradeFlow Docs',[t],'documentation',f'TradeFlow {t} guidance: follow documented controls, collect request identifiers, and verify the current version before acting. This guide is general guidance, not a substitute for specific endpoint documentation.') for i,t in enumerate(TOPICS,1)]
TOK=re.compile(r'[a-z0-9.]+')
def words(x): return TOK.findall(x.lower())
def classify(q):
 q=q.lower()
 if any(x in q for x in ['withdraw','deposit','payment']): return 'payments'
 if any(x in q for x in ['login','password','security','key']): return 'security'
 if any(x in q for x in ['incident','previous','outage','latency']): return 'incident investigation'
 if any(x in q for x in ['order','instrument','market']): return 'trading'
 return 'api troubleshooting' if any(x in q for x in ['api','429','auth','error']) else 'documentation lookup'
def rewrite(q):
 return {'429':'API HTTP 429 rate limit exceeded retry behavior','latency':'API latency incident troubleshooting'}.get(q.lower(),q)
def retrieve(query,filters=None,mode='hybrid_rerank',limit=5):
 start=time.perf_counter(); filters=filters or {}; q=words(query); candidates=[d for d in DOCS if all(str(getattr(d,k,''))==str(v) for k,v in filters.items() if v)]
 df=Counter(w for d in DOCS for w in set(words(d.title+' '+d.content))); avg=sum(len(words(d.content)) for d in DOCS)/len(DOCS); dense=[]; bm=[]
 for d in candidates:
  dw=words(d.title+' '+d.content); c=Counter(dw); dense.append((d,len(set(q)&set(dw))/math.sqrt(max(1,len(set(q))*len(set(dw)))))); bm.append((d,sum(math.log((len(DOCS)-df[t]+.5)/(df[t]+.5)+1)*(c[t]*2.2/(c[t]+1.2*(1-.75+.75*len(dw)/avg))) for t in q)))
 dense.sort(key=lambda x:x[1],reverse=True); bm.sort(key=lambda x:x[1],reverse=True)
 if mode=='dense': ranked=dense
 elif mode=='bm25': ranked=bm
 else:
  rrf=Counter()
  for arr in (dense,bm):
   for rank,(d,_) in enumerate(arr,1): rrf[d.document_id]+=1/(60+rank)
  ranked=[(next(d for d in candidates if d.document_id==i),s) for i,s in rrf.items()]
  if mode=='hybrid_rerank': ranked=[(d,s+.15*len(set(q)&set(words(d.title)))) for d,s in ranked]
  ranked.sort(key=lambda x:x[1],reverse=True)
 return [{'document':d,'score':round(s,4),'passage':d.content} for d,s in ranked[:limit]],round((time.perf_counter()-start)*1000,2)
def generate(q,hits):
 stop={'tradeflow','does','the','a','an','is','are','what','how','why','can','i','my','for','to','of','and','in','with','about','current','previous','version','getting','should','would','could','provide','support','offer','use','check','first','need','compare'}
 informative=set(words(q))-stop
 supported=any(informative & set(words(h['document'].title+' '+h['document'].content)) for h in hits)
 # Refuse novel high-risk concepts even where a generic token (for example "API") matched.
 # Ordinary phrasing is not treated as unknown: "authenticate" should match "authentication".
 unsupported_markers={'quantum','resistant','cryptocurrency','wallet','tax','advice','guarantee','fill'}
 if not hits or not supported or informative & unsupported_markers: return "I couldn't find reliable information about that in the available TradeFlow documentation.",[],True
 l=q.lower(); find=lambda x:next((h for h in hits if x in h['document'].document_id),None)
 if '429' in l or 'rate limit' in l:
  now,old,err=find('rate-v32'),find('rate-v21'),find('errors')
  if ('previous' in l or 'v2' in l) and now and old: return 'The current Orders endpoint allowance is 180 requests/minute in v3.2, compared with 120 requests/minute in v2.1. v3.2 explicitly supersedes v2.1. [1] [2]',[now,old],False
  if now: return 'Your request is likely being rate-limited. The current Orders endpoint limit is 180 requests/minute per account. Inspect Retry-After and X-RateLimit-Remaining, then use exponential backoff with jitter. [1] [2]',[now,err or now],False
 if 'withdraw' in l and find('withdrawals'): return 'Withdrawals require a verified account, two-factor authentication, and a linked payment method in the account holder name. A security hold can apply after a password reset or new bank account. [1]',[find('withdrawals')],False
 if 'reject' in l and find('orders-rejected'): return 'Check instrument status, market hours, buying power, price bands, account permissions, and minimum quantity. Use the rejection code to identify the failed validation. [1]',[find('orders-rejected')],False
 return 'Based on the retrieved TradeFlow documentation: '+hits[0]['passage']+' [1]',[hits[0]],False
def pack(h): return asdict(h['document'])|{'score':h['score'],'passage':h['passage']}

app=FastAPI(title='TradeDocs AI',version='0.2.0'); EVENTS=[]; FEEDBACK=[]; STORE=TelemetryStore(); LLM=OpenAICompatibleGenerator()
class ChatRequest(BaseModel): query:str=Field(min_length=1,max_length=4000); filters:dict[str,str]={}
class Feedback(BaseModel): request_id:str; helpful:bool
@app.get('/health')
def health(): return {'status':'ok','documents':len(DOCS)}
@app.post('/chat')
def chat(req:ChatRequest):
 start=time.perf_counter(); rid=str(uuid.uuid4()); rq=rewrite(req.query); hits,rms=retrieve(rq,req.filters); text,evidence,refused=generate(req.query,hits); generation_ms=0
 if not refused and LLM.configured():
  generated_at=time.perf_counter()
  try:
   candidate=LLM.generate(req.query,evidence); cited=re.findall(r'\[(\d+)\]',candidate)
   if cited and all(1<=int(x)<=len(evidence) for x in cited): text=candidate
  except RuntimeError: pass
  generation_ms=round((time.perf_counter()-generated_at)*1000,2)
 if not refused and any(int(x)>len(evidence) for x in re.findall(r'\[(\d+)\]',text)): raise HTTPException(502,'Citation validation failed')
 event={'request_id':rid,'query':req.query,'intent':classify(req.query),'rewritten_query':rq,'retrieval_ms':rms,'generation_ms':generation_ms,'total_ms':round((time.perf_counter()-start)*1000,2),'retrieved_document_ids':[x['document'].document_id for x in hits],'citations':[x['document'].document_id for x in evidence],'insufficient_evidence':refused}; EVENTS.append(event); STORE.record(event)
 return {'request_id':rid,'answer':text,'intent':event['intent'],'rewritten_query':rq,'evidence':[pack(x) for x in evidence],'retrieved':[pack(x) for x in hits],'insufficient_evidence':refused,'metrics':event}
@app.post('/search')
def search(req:ChatRequest):
 hits,ms=retrieve(req.query,req.filters); return {'results':[pack(x) for x in hits],'retrieval_ms':ms}
@app.get('/documents')
def documents(): return [asdict(x) for x in DOCS]
@app.get('/documents/{document_id}')
def document(document_id:str):
 d=next((x for x in DOCS if x.document_id==document_id),None)
 if not d: raise HTTPException(404,'Document not found')
 return asdict(d)
@app.post('/feedback')
def feedback(item:Feedback):
 FEEDBACK.append(item.model_dump()); STORE.feedback(item.request_id,item.helpful); return {'stored':True}
@app.get('/metrics')
def metrics():
 report=STORE.metrics(); latencies=sorted(x['total_ms'] for x in EVENTS)
 report.update({'p50_latency_ms':latencies[len(latencies)//2] if latencies else 0,'p95_latency_ms':latencies[max(0,math.ceil(len(latencies)*.95)-1)] if latencies else 0,'average_retrieval_ms':round(sum(x['retrieval_ms'] for x in EVENTS)/len(EVENTS),2) if EVENTS else 0,'events':EVENTS[-20:]}); return report
@app.post('/evaluation/run')
def evaluation_run():
 cases=json.loads((Path(__file__).parents[2]/'data'/'evaluation'/'cases.json').read_text())
 result={}
 for mode in ('dense','bm25','hybrid','hybrid_rerank'):
  ranks=[]
  for case in cases:
   if not case['expected_documents']: continue
   ids=[h['document'].document_id for h in retrieve(case['question'],mode=mode)[0]]; expected=set(case['expected_documents']); ranks.append(next((i+1 for i,x in enumerate(ids) if x in expected),None))
  result[mode]={'recall_at_5':round(sum(x is not None for x in ranks)/len(ranks),3),'mrr':round(sum(1/x if x else 0 for x in ranks)/len(ranks),3)}
 return {'cases':len(cases),'retrieval':result}
@app.get('/evaluation/results')
def evaluation_results(): return evaluation_run()
@app.get('/conversations/{conversation_id}')
def conversation(conversation_id:str): return {'conversation_id':conversation_id,'messages':[]}
app.mount('/',StaticFiles(directory=str(Path(__file__).parents[2]/'frontend'),html=True),name='frontend')
