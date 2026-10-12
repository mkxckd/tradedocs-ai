import json

import anyio
from mcp import Client

from app.mcp_server import server


def call(tool: str, args: dict):
    async def run():
        async with Client(server) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            result = await client.call_tool(tool, args)
            return names, json.loads(result.content[0].text)
    return anyio.run(run)


def test_mcp_exposes_read_only_tools_and_answers_with_citations():
    names, data = call("answer_question", {"question": "How often does the WebSocket send heartbeats?"})
    assert set(names) == {"search_documents", "get_document", "answer_question"}
    assert not data["insufficient_evidence"] and data["citations"][0]["document_id"] == "websocket"


def test_mcp_refuses_out_of_scope_questions():
    _, data = call("answer_question", {"question": "Does TradeFlow provide a cryptocurrency wallet?"})
    assert data["insufficient_evidence"] and data["citations"] == []
