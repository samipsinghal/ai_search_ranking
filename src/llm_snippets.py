"""
llm_snippets.py
---------------
Minimal LLM wrapper for query-focused snippets.

Usage:
    from src.llm_snippets import LLMSnippeter
    snip = snippeter.make_snippet(query, passage_text)
"""

import os, time
from typing import Optional

# You can swap this out for any provider; we keep it abstract via a callable.
class LLMSnippeter:
    def __init__(self, call_llm, max_chars: int = 300, timeout_s: float = 10.0):
        """
        call_llm(prompt: str, timeout_s: float) -> str
        max_chars: truncate output for terminal display
        """
        self.call_llm = call_llm
        self.max_chars = max_chars
        self.timeout_s = timeout_s

    def _prompt(self, query: str, passage: str) -> str:
        # Extractive-first, citation-like constraints reduce hallucinations.
        return (
            "You are a snippet generator for a search engine.\n"
            "Goal: Write a concise snippet (2–3 lines) answering the query using ONLY the provided passage text.\n"
            "Rules:\n"
            " - Be faithful: do not introduce facts not present in the passage.\n"
            " - Prefer direct wording from the passage; compress and lightly paraphrase.\n"
            " - Bold occurrences of query terms.\n"
            " - Length: ~30–50 words.\n"
            f"Query: {query}\n"
            "Passage:\n"
            f"\"\"\"\n{passage}\n\"\"\"\n"
            "Snippet:"
        )

    def make_snippet(self, query: str, passage: str) -> str:
        prompt = self._prompt(query, passage)
        text = self.call_llm(prompt, timeout_s=self.timeout_s).strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars - 1] + "…"
        return text


# --- Example provider implementations ---

def openai_provider(model: str = "gpt-4o-mini"):
    """
    Returns a function(prompt, timeout_s) -> str.
    Expects OPENAI_API_KEY in env. Replace with your SDK of choice.
    """
    import os, requests, json
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY for openai_provider()")

    url = "https://api.openai.com/v1/chat/completions"

    def _call(prompt: str, timeout_s: float = 10.0) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 120,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    return _call


def echo_provider():
    """Offline fallback: returns a trimmed first sentence as a 'snippet'."""
    def _call(prompt: str, timeout_s: float = 10.0) -> str:
        # ultra-dumb fallback for demos; never ship as-is.
        passage = prompt.split("Passage:", 1)[-1]
        passage = passage.strip().strip('"').replace("\n", " ")
        return passage[:200]
    return _call
