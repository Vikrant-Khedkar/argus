"""Web search tool backed by ScrapeGraphAI. Exposed via OpenAI function-calling spec."""

import json
from typing import Any

from scrapegraph_py import ScrapeGraphAI

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current factual information. Use when the user asks about "
            "recent events, specific dates, named entities, statistics, or anything you "
            "might not know reliably. Returns extracted facts and source URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "extraction_prompt": {
                    "type": "string",
                    "description": (
                        "What information to extract from the search results, "
                        "e.g. 'Return the exact date and a one-sentence summary.'"
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return. Default 3.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}


def web_search(query: str, extraction_prompt: str | None = None, num_results: int = 3) -> dict[str, Any]:
    sgai = ScrapeGraphAI()
    res = sgai.search(
        query,
        num_results=max(1, min(num_results, 5)),
        prompt=extraction_prompt or "Summarize the key facts answering the query in 2 sentences.",
    )
    if res.status != "success" or not res.data:
        return {"query": query, "error": getattr(res, "error", "search failed"), "results": []}
    return {
        "query": query,
        "results": [{"title": h.title, "url": h.url} for h in (res.data.results or [])],
        "summary": getattr(res.data, "json_data", None),
    }


TOOL_REGISTRY = {"web_search": web_search}


def dispatch(name: str, arguments: dict | str) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = fn(**arguments)
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, default=str)
