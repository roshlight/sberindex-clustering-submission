"""Document parsing and deterministic support checks around model suggestions."""

import hashlib
import io
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from pypdf import PdfReader

from sber_enrichment.office_text import parse_office

OFFICIAL_DOMAINS = (
    "sberindex.ru",
    "sber.ru",
    "sberbank.com",
    "sberbank.ru",
    "s.sber.ru",
    "rosstat.gov.ru",
    "gks.ru",
    "fedstat.ru",
    "nalog.gov.ru",
    "minfin.gov.ru",
    "economy.gov.ru",
    "pravo.gov.ru",
)


def canonical_url(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Missing source URL")
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None:
        raise ValueError("Not a public source URL")
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"gclid", "fbclid"}
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", urlencode(query), "")
    )


def source_id(url):
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:20]


def is_official(url):
    host = urlsplit(url).hostname or ""
    return any(host == domain or host.endswith("." + domain) for domain in OFFICIAL_DOMAINS)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_document(body, content_type, url, max_chars=100_000):
    """Return bounded visible text and observed links; unsupported binary stays binary."""
    content_type = (content_type or "").lower()
    extension = urlsplit(url).path.lower().rsplit(".", 1)[-1]
    if body.startswith(b"PK") and (
        extension in {"xlsx", "docx"} or "officedocument" in content_type
    ):
        if "spreadsheetml" in content_type:
            extension = "xlsx"
        elif "wordprocessingml" in content_type:
            extension = "docx"
        return parse_office(body, "." + extension, max_chars=max_chars)
    links = []
    if body.startswith(b"%PDF") or "pdf" in content_type:
        reader = PdfReader(io.BytesIO(body), strict=False)
        pages = []
        length = 0
        for page in reader.pages[:120]:
            part = page.extract_text() or ""
            pages.append(part)
            length += len(part)
            if length >= max_chars:
                break
        text = "\n".join(pages)
        kind = "pdf"
    elif "html" in content_type or body.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        soup = BeautifulSoup(body, "html.parser")
        for item in soup.find_all("a", href=True):
            try:
                link = canonical_url(urljoin(url, item["href"]))
                if re.search(r"\.(?:pdf|xlsx?|csv|zip|rar|parquet|gpkg)(?:\?|$)", link, re.I):
                    links.append({"url": link, "label": normalize_text(item.get_text(" "))[:200]})
            except ValueError:
                pass
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        kind = "html"
    elif content_type.startswith("text/") or "json" in content_type:
        text = body.decode("utf-8-sig", errors="replace")
        kind = "text"
    else:
        return {"text": "", "kind": "binary", "links": [], "truncated": False}
    return {
        "text": text[:max_chars],
        "kind": kind,
        "links": links,
        "truncated": len(text) > max_chars,
    }


CARD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_id": {"type": "string"},
        "relevance": {"type": "string", "enum": ["high", "medium", "low", "unclear"]},
        "summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    key: {"type": "string"}
                    for key in ["field", "value", "evidence_quote", "limitations"]
                },
                "required": ["field", "value", "evidence_quote", "limitations"],
            },
        },
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "proposed_use": {"type": "string"},
    },
    "required": [
        "source_id",
        "relevance",
        "summary",
        "claims",
        "missing_information",
        "proposed_use",
    ],
}

QUERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "queries": {
            "type": "array",
            "maxItems": 18,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "need_id": {"type": "string", "maxLength": 100},
                    "query": {"type": "string", "maxLength": 250},
                    "purpose": {"type": "string", "maxLength": 300},
                },
                "required": ["need_id", "query", "purpose"],
            },
        }
    },
    "required": ["queries"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "need_id": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string", "enum": ["candidate", "conflict", "missing"]},
                    "reason": {"type": "string"},
                    "next_action": {"type": "string"},
                },
                "required": ["need_id", "source_ids", "status", "reason", "next_action"],
            },
        },
    },
    "required": ["summary", "decisions"],
}


def verify_card(card, expected_source_id, document_text):
    """A quotation check proves textual support, never factual truth or correct joins."""
    if card["source_id"] != expected_source_id:
        raise ValueError("Model substituted the source identifier")
    haystack = normalize_text(document_text)
    claims = []
    for claim in card["claims"]:
        quote = normalize_text(claim["evidence_quote"])
        supported = len(quote) >= 12 and quote in haystack
        claims.append(
            {
                **claim,
                "quote_match": supported,
                "normalized_text_offset": haystack.find(quote) if supported else None,
                "status": "candidate_for_review" if supported else "rejected_no_quote",
            }
        )
    return {
        **card,
        "claims": claims,
        "authority": "model_assisted_candidate",
        "automatic_numeric_imputation": False,
    }
