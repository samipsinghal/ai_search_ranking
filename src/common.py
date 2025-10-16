"""
common.py
----------
Utilities for text parsing and tokenization.
Removes HTML markup, lowercases text, and yields alphanumeric terms.
"""

import re
from html.parser import HTMLParser

# Precompiled regex: match contiguous sequences of A–Z, a–z, or digits.
_TOKEN = re.compile(r"[A-Za-z0-9]+")

class _HTMLStripper(HTMLParser):
    """Simple HTML stripper that ignores <script> and <style> tags."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts)

def strip_html(html_text: str) -> str:
    """Return plain text with HTML markup removed."""
    stripper = _HTMLStripper()
    try:
        stripper.feed(html_text)
        return stripper.get_text()
    except Exception:
        # In case of malformed HTML, just return the original string.
        return html_text

def tokenize(text: str):
    """Yield lowercase alphanumeric tokens from the given text."""
    clean = strip_html(text).lower()
    for match in _TOKEN.finditer(clean):
        yield match.group(0)
