from __future__ import annotations

import re
import string

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("<br />", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE_RE.sub(" ", text)
    return text
