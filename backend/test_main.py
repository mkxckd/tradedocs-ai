import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'backend'))
from app.main import retrieve,generate,DOCS
from app.main import app
from fastapi.testclient import TestClient

def test_corpus_has_fifty_plus_documents():
    assert len(DOCS)>=50

def test_version_aware_retrieval():
    hits,_=retrieve('current API rate limit',mode='hybrid_rerank')
    assert any(x['document'].document_id=='api-rate-v32' for x in hits)

def test_insufficient_evidence():
    text,_,refused=generate('quantum-resistant API keys',[])
    assert refused and 'couldn' in text

def test_metadata_filter():
    hits,_=retrieve('rate limit',{'version':'2.1'})
    assert hits and all(x['document'].version=='2.1' for x in hits)

def test_hybrid_returns_ranked_evidence():
    hits,_=retrieve('HTTP 429 retry rate limit',mode='hybrid_rerank')
    assert len(hits)==5 and hits[0]['score'] >= hits[-1]['score']

def test_unknown_feature_refuses_even_with_generic_api_terms():
    _,evidence,refused=generate('Does TradeFlow support quantum-resistant API keys?',retrieve('Does TradeFlow support quantum-resistant API keys?')[0])
    assert refused and not evidence

def test_version_conflict_response_contains_both_sources():
    text,evidence,refused=generate('Compare current rate limits with previous v2 version',retrieve('Compare current rate limits with previous v2 version')[0])
    assert not refused and '180' in text and '120' in text and len(evidence)==2

def test_health_endpoint():
    response=TestClient(app).get('/health')
    assert response.status_code==200 and response.json()['documents']>=50

def test_chat_endpoint_and_citations():
    response=TestClient(app).post('/chat',json={'query':'Why am I getting a 429 error?'})
    data=response.json()
    assert response.status_code==200 and data['evidence'] and '[1]' in data['answer']

def test_empty_query_rejected():
    assert TestClient(app).post('/chat',json={'query':''}).status_code==422

def test_document_endpoint_404():
    assert TestClient(app).get('/documents/not-real').status_code==404

def test_feedback_endpoint():
    client=TestClient(app); response=client.post('/feedback',json={'request_id':'test-feedback','helpful':True})
    assert response.status_code==200 and response.json()['stored']
