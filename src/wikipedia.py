from __future__ import annotations

import html
import re
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.stem import PorterStemmer as _PorterStemmer

    try:
        _STOP_WORDS = set(_nltk_stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOP_WORDS = set(_nltk_stopwords.words("english"))
    _STEMMER = _PorterStemmer()
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    _STOP_WORDS = set()
    _STEMMER = None


@dataclass
class SectionHit:
    title: str
    pageid: int
    url: str
    section_title: str
    text: str  # full section text

    bm25: float = 0.0
    dense: float = 0.0
    score: float = 0.0
    best_chunk: str = ""


class WikipediaHybridSectionRetriever:
    _SKIP_SECTIONS = {
        "references", "see also", "external links", "further reading",
        "bibliography", "notes", "footnotes", "notes and references",
        "citations", "sources", "gallery", "awards", "discography", "filmography",
    }

    def __init__(
        self,
        emb_model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        chunk_overlap: int = 16,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        user_agent: str = "WikipediaHybridSectionRetriever",
        lang: str = "en",
        session: Optional[requests.Session] = None,
    ):
        self.lang = lang
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": f"{user_agent} (gladkykh.sviatoslav@gmail.com; AI Research Bot)"}
        self.sess = session or requests.Session()

        self.embedder = SentenceTransformer(emb_model_name)
        self.max_passage_tokens = self.embedder.max_seq_length
        self.tok = self.embedder.tokenizer
        # Must be computed before overriding model_max_length below
        self.num_special_tokens = len(self.tok.encode("", add_special_tokens=True))
        # Suppress tokenizer length warnings; chunking enforces the limit manually
        self.tok.model_max_length = int(1e9)

        self.chunk_overlap = chunk_overlap
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    # ------------------- Public API -------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        page_limit: int = 7,
        section_limit_per_page: int = 15,
    ) -> List[SectionHit]:
        """search → fetch → split → clean → filter → chunk → score → top-k"""
        pages = self._search_pages(query, limit=page_limit)
        if not pages:
            return []

        sections: List[SectionHit] = []
        for p in pages:
            title, pageid = p["title"], p["pageid"]
            url = f"https://{self.lang}.wikipedia.org/?curid={pageid}"
            html_blob = self._fetch_page_html(title)
            secs = self._split_html_into_sections(html_blob)[:section_limit_per_page]

            for sec_title, sec_html in secs:
                if sec_title.strip().lower() in self._SKIP_SECTIONS:
                    continue
                text = self._html_to_text(sec_html)
                if len(text.strip()) >= 50:
                    sections.append(
                        SectionHit(title=title, pageid=pageid, url=url, section_title=sec_title, text=text)
                    )

        if not sections:
            return []

        self._score_sections(query, sections)
        sections.sort(key=lambda x: x.score, reverse=True)
        return sections[:k]

    # ------------------- Wikipedia API -------------------

    def _get_with_retry(self, params: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                r = self.sess.get(self.api, params=params, headers=self.headers, timeout=20)

                if r.status_code == 429:
                    delay = (base_delay * (2**attempt)) + random.uniform(0, 1)
                    print(f"Rate limited (429). Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue

                r.raise_for_status()
                return r.json()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                delay = (base_delay * (2**attempt)) + random.uniform(0, 1)
                time.sleep(delay)

        return {}

    def _search_pages(self, query: str, limit: int) -> List[Dict[str, Any]]:
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
        data = self._get_with_retry(params)
        return [{"pageid": x["pageid"], "title": x["title"]} for x in data.get("query", {}).get("search", [])]

    def _fetch_page_html(self, title: str) -> str:
        params = {"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1}
        data = self._get_with_retry(params)
        return data.get("parse", {}).get("text", {}).get("*", "")

    # ------------------- HTML → sections -------------------

    def _split_html_into_sections(self, page_html: str) -> List[tuple[str, str]]:
        """Returns (title, body_html) pairs. Content before the first heading is labelled 'Lead'."""
        heading_re = re.compile(r"(?is)<h[2-6][^>]*>.*?</h[2-6]>")
        headings = list(heading_re.finditer(page_html))

        if not headings:
            return [("Lead", page_html)]

        out: List[tuple[str, str]] = [("Lead", page_html[: headings[0].start()])]

        for i, m in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(page_html)
            title = self._heading_text(m.group(0)) or f"Section {i + 1}"
            out.append((title, page_html[m.end() : end]))

        return out

    def _heading_text(self, heading_html: str) -> str:
        s = re.sub(r"(?is)<.*?>", " ", heading_html)
        s = html.unescape(s)
        return re.sub(r"\s+", " ", s).strip()

    # ------------------- Scoring -------------------

    def _score_sections(self, query: str, sections: List[SectionHit]) -> None:
        # Flatten all chunks, tracking which section each chunk belongs to
        chunk_texts: List[str] = []
        chunk_to_sec: List[int] = []

        for si, sec in enumerate(sections):
            for ch in self._chunk_text(sec.text):
                chunk_texts.append(ch)
                chunk_to_sec.append(si)

        bm25 = BM25Okapi([self._bm25_tokens(t) for t in chunk_texts])
        bm25_scores = bm25.get_scores(self._bm25_tokens(query)).astype(np.float32)

        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        chunk_vecs = self.embedder.encode(chunk_texts, normalize_embeddings=True)
        dense_scores = (chunk_vecs @ q_vec).astype(np.float32)

        # Section score = max over its chunks; best-matching chunk is stored for display
        sec_bm25 = np.full(len(sections), -np.inf, dtype=np.float32)
        sec_dense = np.full(len(sections), -np.inf, dtype=np.float32)
        best_chunk_idx = np.full(len(sections), -1, dtype=np.int32)

        for i, si in enumerate(chunk_to_sec):
            if bm25_scores[i] > sec_bm25[si]:
                sec_bm25[si] = bm25_scores[i]
            if dense_scores[i] > sec_dense[si]:
                sec_dense[si] = dense_scores[i]
                best_chunk_idx[si] = i

        sec_bm25_n = self._minmax(sec_bm25)
        sec_dense_n = self._minmax(sec_dense)
        sec_score = self.bm25_weight * sec_bm25_n + self.dense_weight * sec_dense_n

        for i, sec in enumerate(sections):
            sec.bm25 = float(sec_bm25_n[i])
            sec.dense = float(sec_dense_n[i])
            sec.score = float(sec_score[i])
            j = int(best_chunk_idx[i])
            if j >= 0:
                sec.best_chunk = chunk_texts[j]

    # ------------------- Chunking -------------------

    def _chunk_text(self, text: str) -> List[str]:
        # -3 safety margin: character-offset slicing at subword boundaries can expand
        # by 1-2 tokens when the fragment is re-tokenised with special tokens added
        effective_max = self.max_passage_tokens - self.num_special_tokens - 3
        return self._split_by_tokens(text.strip(), effective_max, self.chunk_overlap) or [""]

    def _split_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        # Slice the original text by character offsets rather than decoding token ids —
        # decoding an uncased tokeniser corrupts capitalisation
        encoding = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        if len(ids) <= max_tokens:
            return [text]

        step = max(1, max_tokens - overlap)
        chunks = []
        for start in range(0, len(ids), step):
            end = min(start + max_tokens, len(ids))
            chunk = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(ids):
                break
        return chunks or [text[:200]]

    # ------------------- Text utils -------------------

    def _bm25_tokens(self, s: str) -> List[str]:
        tokens = re.findall(r"\w+", s.lower())
        if _NLTK_AVAILABLE:
            tokens = [t for t in tokens if t not in _STOP_WORDS]
            tokens = [_STEMMER.stem(t) for t in tokens]
        return tokens

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi == lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _html_to_text(self, s: str) -> str:
        soup = BeautifulSoup(s, "html.parser")

        bad_classes = re.compile(
            r"(navbox|infobox|sidebar|metadata|mw-editsection|reference|noprint|hatnote|thumb|figure|image)"
        )
        for tag in soup.find_all(class_=bad_classes):
            tag.decompose()
        for tag in soup.find_all(role="navigation"):
            tag.decompose()
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        for math_tag in soup.find_all("math"):
            annotation = math_tag.find("annotation", encoding=re.compile(r"tex", re.I))
            if annotation:
                latex = annotation.get_text(strip=True)
                if latex.startswith(r"{\displaystyle") and latex.endswith("}"):
                    latex = latex[15:-1].strip()
                    math_tag.replace_with(f" $${latex}$$ ")
                else:
                    math_tag.replace_with(f" ${latex}$ ")

        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]):
            tag.append("\n")
        for tag in soup.find_all("br"):
            tag.replace_with("\n")

        text = html.unescape(soup.get_text())
        text = text.replace("\u2060", "")  # Wikipedia inserts zero-width joiners

        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[note \d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^\d+\s*$", "", text)           # lone citation digit lines
        text = re.sub(r"\bv\s+t\s+e\b", "", text)          # navbox artifact
        text = re.sub(r"(?m)^(read|edit|view history|talk|contributions)\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Coordinates:.*?(?=\n|$)", "", text)
        text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.IGNORECASE)

        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
