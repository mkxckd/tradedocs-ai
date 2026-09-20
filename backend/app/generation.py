"""Generation adapter. The default is intentionally evidence-only and deterministic."""
import json, os
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from typing import Protocol

class Generator(Protocol):
    def generate(self, query: str, context: list[dict]) -> str: ...

class EvidenceOnlyGenerator:
    """A safe local fallback: it never introduces knowledge outside supplied evidence."""
    def generate(self, query: str, context: list[dict]) -> str:
        if not context: return "I couldn't find reliable information about that in the available TradeFlow documentation."
        return f"Based on the retrieved TradeFlow documentation: {context[0]['passage']} [1]"

class OpenAICompatibleGenerator:
    """Optional adapter for a configured OpenAI-compatible endpoint; never required for tests."""
    def __init__(self):
        self.api_key=os.getenv('LLM_API_KEY'); self.base_url=os.getenv('LLM_BASE_URL','https://api.openai.com/v1'); self.model=os.getenv('LLM_MODEL','gpt-4.1-mini')
    def configured(self): return bool(self.api_key)
    def generate(self, query: str, context: list[dict]) -> str:
        if not self.configured(): raise RuntimeError('LLM_API_KEY is not configured')
        sources='\n'.join(f"[{i+1}] {item['document'].title}: {item['passage']}" for i,item in enumerate(context))
        prompt=("Answer only from the provided TradeFlow evidence. Cite every factual sentence with [n]. "
                "If evidence is insufficient, say so exactly and do not guess.\n\nEvidence:\n"+sources+"\n\nQuestion: "+query)
        payload=json.dumps({'model':self.model,'messages':[{'role':'system','content':'You are a strictly evidence-grounded support assistant.'},{'role':'user','content':prompt}],'temperature':0}).encode()
        request=Request(self.base_url.rstrip('/')+'/chat/completions',data=payload,headers={'Authorization':'Bearer '+self.api_key,'Content-Type':'application/json'},method='POST')
        try:
            with urlopen(request,timeout=15) as response: return json.loads(response.read())['choices'][0]['message']['content']
        except (HTTPError,URLError,TimeoutError,KeyError,ValueError) as exc:
            raise RuntimeError(f'LLM generation failed: {type(exc).__name__}') from exc
