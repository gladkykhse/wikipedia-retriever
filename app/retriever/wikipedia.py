import asyncio
import html
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
import numpy as np
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
    text: str

    bm25: float = 0.0
    dense: float = 0.0
    score: float = 0.0
    best_chunk: str = ""


@dataclass
class _PageSections:
    """Scored-agnostic per-page payload; shared across queries in a batch."""

    title: str
    pageid: int
    url: str
    # list of (section_title, section_text)
    sections: list[tuple[str, str]] = field(default_factory=list)


class WikipediaHybridSectionRetriever:
    _SKIP_SECTIONS = {
        "references",
        "see also",
        "external links",
        "further reading",
        "bibliography",
        "notes",
        "footnotes",
        "notes and references",
        "citations",
        "sources",
        "gallery",
        "awards",
        "discography",
        "filmography",
    }

    def __init__(
        self,
        emb_model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        chunk_overlap: int = 16,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        user_agent: str = "WikipediaHybridSectionRetriever",
        lang: str = "en",
        client: Optional[httpx.AsyncClient] = None,
        http_concurrency: int = 8,
        on_page_fetched: Optional[Callable[[], None]] = None,
        on_cache_hit: Optional[Callable[[], None]] = None,
    ):
        self.lang = lang
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.headers = {
            "User-Agent": user_agent,
        }
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=20.0,
            http2=True,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers=self.headers,
        )
        self._sem = asyncio.Semaphore(http_concurrency)

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

        self._on_page_fetched = on_page_fetched
        self._on_cache_hit = on_cache_hit

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    # ------------------- Public API -------------------

    async def retrieve(
        self,
        queries: list[str],
        k: int = 5,
        page_limit: int = 7,
        section_limit_per_page: int = 15,
    ) -> list[list[SectionHit]]:
        """search → fetch (concurrent + deduped) → split → clean → filter → chunk → score → top-k.

        Always list-in, list-out: returns one result list per input query, in order.
        """
        if not queries:
            return []

        # 1) Search all queries concurrently, collect per-query page orderings.
        search_results = await asyncio.gather(
            *[self._search_pages(q, limit=page_limit) for q in queries]
        )
        per_query_pageids: list[list[int]] = [
            [p["pageid"] for p in result] for result in search_results
        ]

        # 2) Build the union of pages across the batch; fetch each exactly once.
        unique: dict[int, str] = {}
        for result in search_results:
            for p in result:
                unique.setdefault(p["pageid"], p["title"])

        page_cache: dict[int, _PageSections] = {}
        async def _fetch_one(pageid: int, title: str) -> None:
            html_blob = await self._fetch_page_html(title)
            url = f"https://{self.lang}.wikipedia.org/?curid={pageid}"

            def _parse() -> list[tuple[str, str]]:
                secs_raw = self._split_html_into_sections(html_blob)[:section_limit_per_page]
                sections: list[tuple[str, str]] = []
                for sec_title, sec_html in secs_raw:
                    if sec_title.strip().lower() in self._SKIP_SECTIONS:
                        continue
                    text = self._html_to_text(sec_html)
                    if len(text.strip()) >= 50:
                        sections.append((sec_title, text))
                return sections

            sections = await asyncio.to_thread(_parse)
            page_cache[pageid] = _PageSections(
                title=title, pageid=pageid, url=url, sections=sections
            )

        await asyncio.gather(*[_fetch_one(pid, title) for pid, title in unique.items()])

        # 3) Build per-query SectionHit lists from the cache (fresh copies; scoring mutates).
        per_query_sections: list[list[SectionHit]] = []
        for pageids in per_query_pageids:
            hits: list[SectionHit] = []
            for pid in pageids:
                pc = page_cache.get(pid)
                if pc is None:
                    continue
                for sec_title, text in pc.sections:
                    hits.append(
                        SectionHit(
                            title=pc.title,
                            pageid=pc.pageid,
                            url=pc.url,
                            section_title=sec_title,
                            text=text,
                        )
                    )
            per_query_sections.append(hits)

        # 4) Score each query independently, batching embeddings across the whole batch.
        await asyncio.to_thread(self._score_all, queries, per_query_sections)

        # 5) Sort and truncate per query.
        out: list[list[SectionHit]] = []
        for hits in per_query_sections:
            hits.sort(key=lambda x: x.score, reverse=True)
            out.append(hits[:k])
        return out

    # ------------------- Wikipedia API -------------------

    async def _get_with_retry(self, params: dict[str, Any], max_retries: int = 5) -> dict[str, Any]:
        base_delay = 1.0
        async with self._sem:
            for attempt in range(max_retries):
                try:
                    r = await self.client.get(self.api, params=params)
                    if r.status_code == 429:
                        delay = (base_delay * (2**attempt)) + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                        continue
                    r.raise_for_status()
                    return r.json()
                except httpx.HTTPError:
                    if attempt == max_retries - 1:
                        raise
                    delay = (base_delay * (2**attempt)) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
        return {}

    async def _search_pages(self, query: str, limit: int) -> list[dict[str, Any]]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        data = await self._get_with_retry(params)
        return [
            {"pageid": x["pageid"], "title": x["title"]}
            for x in data.get("query", {}).get("search", [])
        ]

    async def _fetch_page_html(self, title: str) -> str:
        params = {"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1}
        data = await self._get_with_retry(params)
        if self._on_page_fetched is not None:
            self._on_page_fetched()
        return data.get("parse", {}).get("text", {}).get("*", "")

    # ------------------- HTML → sections -------------------

    def _split_html_into_sections(self, page_html: str) -> list[tuple[str, str]]:
        """Returns (title, body_html) pairs. Content before the first heading is labelled 'Lead'."""
        heading_re = re.compile(r"(?is)<h[2-6][^>]*>.*?</h[2-6]>")
        headings = list(heading_re.finditer(page_html))

        if not headings:
            return [("Lead", page_html)]

        out: list[tuple[str, str]] = [("Lead", page_html[: headings[0].start()])]
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

    def _score_all(
        self,
        queries: list[str],
        per_query_sections: list[list[SectionHit]],
    ) -> None:
        """Score every query's sections. Batches all encoding into one call."""
        # Collect chunks per query, deduplicating chunk texts across the whole batch.
        per_query_chunks: list[list[str]] = []
        per_query_chunk_to_sec: list[list[int]] = []

        all_chunks: list[str] = []
        chunk_text_to_idx: dict[str, int] = {}

        for sections in per_query_sections:
            chunk_texts: list[str] = []
            chunk_to_sec: list[int] = []
            for si, sec in enumerate(sections):
                for ch in self._chunk_text(sec.text):
                    chunk_texts.append(ch)
                    chunk_to_sec.append(si)
                    if ch not in chunk_text_to_idx:
                        chunk_text_to_idx[ch] = len(all_chunks)
                        all_chunks.append(ch)
            per_query_chunks.append(chunk_texts)
            per_query_chunk_to_sec.append(chunk_to_sec)

        # Single batched embedding call for queries + all unique chunks.
        if all_chunks or queries:
            # Larger batch sizes amortise per-batch overhead on CPU; 128 is close to
            # the sweet spot for MiniLM on commodity cores without overshooting memory.
            encoded = self.embedder.encode(
                list(queries) + all_chunks,
                batch_size=128,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            q_vecs = np.asarray(encoded[: len(queries)], dtype=np.float32)
            all_chunk_vecs = np.asarray(encoded[len(queries) :], dtype=np.float32)
        else:
            q_vecs = np.zeros((0, 384), dtype=np.float32)
            all_chunk_vecs = np.zeros((0, 384), dtype=np.float32)

        for qi, sections in enumerate(per_query_sections):
            chunk_texts = per_query_chunks[qi]
            chunk_to_sec = per_query_chunk_to_sec[qi]
            if not sections or not chunk_texts:
                continue

            # BM25 is cheap and per-query-corpus; rebuild per query.
            bm25 = BM25Okapi([self._bm25_tokens(t) for t in chunk_texts])
            bm25_scores = bm25.get_scores(self._bm25_tokens(queries[qi])).astype(np.float32)

            chunk_vec_idx = np.array(
                [chunk_text_to_idx[t] for t in chunk_texts], dtype=np.int64
            )
            chunk_vecs = all_chunk_vecs[chunk_vec_idx]
            dense_scores = (chunk_vecs @ q_vecs[qi]).astype(np.float32)

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

    def _chunk_text(self, text: str) -> list[str]:
        # -3 safety margin: character-offset slicing at subword boundaries can expand
        # by 1-2 tokens when the fragment is re-tokenised with special tokens added
        effective_max = self.max_passage_tokens - self.num_special_tokens - 3
        return self._split_by_tokens(text.strip(), effective_max, self.chunk_overlap) or [""]

    def _split_by_tokens(self, text: str, max_tokens: int, overlap: int) -> list[str]:
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

    def _bm25_tokens(self, s: str) -> list[str]:
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
        text = text.replace("⁠", "")  # Wikipedia inserts zero-width joiners

        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[note \d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^\d+\s*$", "", text)
        text = re.sub(r"\bv\s+t\s+e\b", "", text)
        text = re.sub(r"(?m)^(read|edit|view history|talk|contributions)\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Coordinates:.*?(?=\n|$)", "", text)
        text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.IGNORECASE)

        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
