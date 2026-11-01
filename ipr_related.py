"""Relevance-aware internal link suggestions: which pages should link to a target, and with what anchor.

Two ways to measure "related":

  * Embeddings (best). Upload a CSV of page embeddings made with a local model such as EmbeddingGemma.
    Pages with similar meaning have a high cosine similarity.
  * Text (TF-IDF, no setup). Compares the words (single words and two-word phrases) in each page's
    title, H1, meta description and, if you supply it, body text.

Relevance decides who is eligible to link. Link strength (the existing strength-per-outlink measure)
only decides the order among relevant pages, so a strong but unrelated page is never suggested.

Anchor text comes from the page text: the tool looks for a phrase that describes the target
(its H1, its title, or its most distinctive phrases) inside the linking page's body and suggests
that exact wording. Suggestions are for a person to review, not to apply blindly.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

import ipr_core as ipr

_URL_NAMES = ("Address", "URL", "Url", "Page")
_BODY_NAMES = ("Text", "Body", "Body Text", "Content", "Page Text", "Main Content", "Extracted Text", "Copy")
_TITLE_NAMES = ("Title 1", "Title")
_H1_NAMES = ("H1-1", "H1 1", "H1")
_META_NAMES = ("Meta Description 1", "Meta Description")

_STOP = frozenset("""a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up us very was we were what when where which while who whom why will with would you
your yours yourself yourselves one two get got let may might must shall also per via
browse shop buy see view read find explore discover click learn free new""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_MAX_CHARS = 20000


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def read_embeddings_csv(file_or_path) -> pd.DataFrame:
    """Read an embeddings CSV. Vector columns stay numeric, so a 768-wide file does not balloon in memory."""
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    df = pd.read_csv(file_or_path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_packed(s) -> Optional[np.ndarray]:
    """'[0.1, 0.2, ...]' or '0.1,0.2,...' or '0.1 0.2 ...' to a float vector."""
    if not isinstance(s, str):
        return None
    s = s.strip().strip("[]()")
    if not s:
        return None
    try:
        return np.array([float(p) for p in re.split(r"[,\s;]+", s) if p], dtype=np.float32)
    except ValueError:
        return None


@dataclass
class VectorIndex:
    """Cosine similarity over page embeddings (rows are unit length)."""
    urls: List[str]
    matrix: np.ndarray
    method: str = "Embeddings (cosine similarity)"
    dims: int = 0
    dropped: int = 0
    pos: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.pos = {u: i for i, u in enumerate(self.urls)}
        self.dims = int(self.matrix.shape[1]) if len(self.urls) else 0

    def has(self, url: str) -> bool:
        return url in self.pos

    def sim(self, url: str) -> np.ndarray:
        return self.matrix @ self.matrix[self.pos[url]]


def load_embeddings(df: pd.DataFrame, *, ignore_host: bool = False) -> VectorIndex:
    """Build a VectorIndex from a table with an Address column plus either one numeric column per
    dimension or one column holding the whole vector as text (JSON-style list or comma separated)."""
    ucol = ipr._col(df, *_URL_NAMES)
    if ucol is None:
        raise ValueError("The embeddings file needs an Address (or URL) column.")
    others = [c for c in df.columns if c != ucol]
    if not others:
        raise ValueError("The embeddings file has an Address column but no vector columns.")
    packed = next((c for c in others if c.lower() in ("embedding", "embeddings", "vector", "vectors")), None)
    vecs: List[Optional[np.ndarray]]
    if packed or len(others) == 1:
        col = packed or others[0]
        vecs = [_parse_packed(v) for v in df[col]]
    else:
        num = df[others].apply(pd.to_numeric, errors="coerce")
        num = num.loc[:, num.notna().any()]                 # drop non-vector columns such as a title
        if num.shape[1] < 2:
            raise ValueError("Could not find numeric vector columns in the embeddings file.")
        ok = num.notna().all(axis=1).to_numpy()
        arr = num.to_numpy(dtype=np.float32)
        vecs = [arr[i] if ok[i] else None for i in range(len(arr))]
    lens = Counter(len(v) for v in vecs if v is not None)
    if not lens:
        raise ValueError("No usable vectors were found in the embeddings file.")
    dim = lens.most_common(1)[0][0]
    urls, rows, seen, dropped = [], [], set(), 0
    for u, v in zip(df[ucol].astype(str), vecs):
        nu = ipr.norm_url(u, ignore_host=ignore_host)
        if v is None or len(v) != dim or not nu or nu in seen or not np.isfinite(v).all():
            dropped += 1
            continue
        n = float(np.linalg.norm(v))
        if n == 0:
            dropped += 1
            continue
        seen.add(nu)
        urls.append(nu)
        rows.append(v / n)
    if not urls:
        raise ValueError("No usable vectors were found in the embeddings file.")
    return VectorIndex(urls, np.vstack(rows).astype(np.float32), dropped=dropped)


def _pick(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    return ipr._col(df, *names)


def build_page_text(internal: Optional[pd.DataFrame] = None, text: Optional[pd.DataFrame] = None,
                    *, ignore_host: bool = False) -> pd.DataFrame:
    """One row per page with columns url, title, h1, meta, body, from the Internal: All export
    (Title 1, H1-1, Meta Description 1) and/or a page text CSV (Address plus a body text column)."""
    parts = []
    for df in (internal, text):
        if df is None or not len(df):
            continue
        ucol = _pick(df, _URL_NAMES)
        if ucol is None:
            continue
        out = pd.DataFrame({"url": df[ucol].astype(str).map(lambda u: ipr.norm_url(u, ignore_host=ignore_host))})
        for key, names in (("title", _TITLE_NAMES), ("h1", _H1_NAMES), ("meta", _META_NAMES), ("body", _BODY_NAMES)):
            c = _pick(df, names)
            out[key] = df[c].fillna("").astype(str).to_numpy() if c else ""
        if df is text and not _pick(df, _BODY_NAMES):
            extra = [c for c in df.columns if c != ucol and c not in
                     [_pick(df, _TITLE_NAMES), _pick(df, _H1_NAMES), _pick(df, _META_NAMES)]]
            ext = [c for c in extra if c.lower().startswith("extract") or "extraction" in c.lower()]
            cand = ext or extra
            if cand:                                         # choose the column with the longest values
                best = max(cand, key=lambda c: df[c].fillna("").astype(str).str.len().mean())
                out["body"] = df[best].fillna("").astype(str).to_numpy()
        parts.append(out[out.url != ""])
    if not parts:
        return pd.DataFrame(columns=["url", "title", "h1", "meta", "body"])
    allp = pd.concat(parts, ignore_index=True)
    agg = allp.groupby("url", sort=False).agg(lambda s: max(s, key=len) if len(s) else "").reset_index()
    return agg[["url", "title", "h1", "meta", "body"]]


def read_anchor_table(file_or_path) -> pd.DataFrame:
    """Destination and Anchor of followed links from an All Inlinks export (small, only 3 columns kept)."""
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    keep = {"destination", "anchor", "follow", "type"}
    df = pd.read_csv(file_or_path, dtype=str, low_memory=False, encoding="utf-8-sig", keep_default_na=False,
                     usecols=lambda c: c.strip().lower() in keep)
    df.columns = [c.strip() for c in df.columns]
    return df


def anchor_counts(inlinks: Optional[pd.DataFrame]) -> Dict[str, Counter]:
    """target -> Counter of the lower-cased anchors used by followed links that point at it."""
    if inlinks is None or not len(inlinks):
        return {}
    dc, ac = ipr._col(inlinks, "destination"), ipr._col(inlinks, "anchor")
    if not (dc and ac):
        return {}
    d = inlinks
    fc, tc = ipr._col(d, "follow"), ipr._col(d, "type")
    if fc:
        d = d[d[fc].astype(str).str.lower().isin(["true", "1", "yes"])]
    if tc:
        d = d[d[tc].astype(str).str.lower().isin(["hyperlink", "", "a"])]
    d = d[d[ac].astype(str).str.strip() != ""]
    out: Dict[str, Counter] = {}
    for dest, anc in zip(d[dc].astype(str), d[ac].astype(str)):
        out.setdefault(ipr.norm_url(dest), Counter())[anc.strip().lower()] += 1
    return out


# --------------------------------------------------------------------------------------
# TF-IDF (no model needed)
# --------------------------------------------------------------------------------------

_CLAUSE = re.compile(r"[.,;:!?()\[\]|\u2013\u2014\n\r]+|\s-\s")


def _terms(text: str) -> List[str]:
    """Unigrams and adjacent two-word phrases (within one clause, no stop words), so phrases appear verbatim in the text."""
    out: List[str] = []
    for clause in _CLAUSE.split(text.lower()):
        prev = None
        for t in _TOKEN.findall(clause):
            ok = t not in _STOP and len(t) > 1 and not t.isdigit()
            if ok:
                out.append(t)
                if prev is not None:
                    out.append(prev + " " + t)
            prev = t if ok else None
    return out


def _doc_text(row) -> str:
    # title and H1 are repeated so they weigh more than a body paragraph
    return " ".join([row.title, row.title, row.h1, row.h1, row.meta, row.body[:_MAX_CHARS]])


@dataclass
class TextIndex:
    """TF-IDF vectors (words and two-word phrases) over page text, cosine similarity."""
    urls: List[str]
    matrix: sparse.csr_matrix
    idf: Dict[str, float]
    pages: pd.DataFrame
    method: str = "Text (TF-IDF, words and phrases)"
    pos: Dict[str, int] = field(default_factory=dict)
    _text_pos: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.pos = {u: i for i, u in enumerate(self.urls)}
        self._text_pos = {u: i for i, u in enumerate(self.pages.url)}

    def has(self, url: str) -> bool:
        return url in self.pos

    def sim(self, url: str) -> np.ndarray:
        i = self.pos[url]
        return np.asarray((self.matrix @ self.matrix[i].T).todense()).ravel()

    def page(self, url: str) -> Optional[pd.Series]:
        i = self._text_pos.get(url)
        return None if i is None else self.pages.iloc[i]

    def keyphrases(self, url: str, n: int = 8) -> List[str]:
        """The page's most distinctive words and phrases (TF-IDF weight), phrases favoured over single words."""
        p = self.page(url)
        if p is None:
            return []
        tf = Counter(_terms(_doc_text(p)))
        scored = []
        for term, c in tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            w = (1 + math.log(c)) * idf * (1.3 if " " in term else 1.0)
            scored.append((w, term))
        scored.sort(reverse=True)
        picked: List[str] = []
        for _, term in scored:
            if any(term in q or q in term for q in picked):
                continue
            picked.append(term)
            if len(picked) >= n:
                break
        return picked


def build_text_index(pages: pd.DataFrame, *, min_df: int = 2, max_df_ratio: float = 0.5) -> TextIndex:
    """TF-IDF index from build_page_text(). Needs at least two pages with some text."""
    pages = pages.reset_index(drop=True)
    docs = [Counter(_terms(_doc_text(r))) for r in pages.itertuples()]
    keep = [i for i, d in enumerate(docs) if d]
    if len(keep) < 2:
        raise ValueError("Not enough page text to compare. Supply Title, H1, meta description or body text.")
    n = len(keep)
    df_count: Counter = Counter()
    for i in keep:
        df_count.update(docs[i].keys())
    idf_all = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df_count.items()}
    vocab = {t: j for j, t in enumerate(t for t, c in df_count.items() if c >= min_df and c <= max(2, max_df_ratio * n))}
    rows, cols, vals = [], [], []
    for r, i in enumerate(keep):
        for t, c in docs[i].items():
            j = vocab.get(t)
            if j is not None:
                rows.append(r)
                cols.append(j)
                vals.append((1 + math.log(c)) * idf_all[t])
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(n, max(len(vocab), 1)), dtype=np.float32)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    m = sparse.diags(1.0 / norms) @ m
    return TextIndex([pages.url[i] for i in keep], m.tocsr(), idf_all, pages)


# --------------------------------------------------------------------------------------
# Anchor text
# --------------------------------------------------------------------------------------

def _clean_title(t: str) -> str:
    return re.split(r"\s+[|–—\-:]\s+", (t or "").strip())[0].strip()


def _sentence_around(body: str, start: int, end: int, limit: int = 220) -> str:
    left = max(body.rfind(". ", 0, start), body.rfind("\n", 0, start))
    left = 0 if left < 0 else left + 1
    m = re.search(r"[.!?](\s|$)", body[end:])
    right = len(body) if not m else end + m.end()
    s = body[left:right].strip()
    return s if len(s) <= limit else "..." + body[max(start - limit // 2, left):min(end + limit // 2, right)].strip() + "..."


def suggest_anchor(text_index: Optional[TextIndex], source: str, target: str) -> Dict[str, str]:
    """Anchor text for source -> target: a phrase that describes the target and already appears in the source body."""
    if text_index is None:
        return dict(suggested_anchor="", anchor_basis="No page text supplied", context="")
    tp, sp = text_index.page(target), text_index.page(source)
    if tp is None:
        return dict(suggested_anchor="", anchor_basis="No text for the target page", context="")
    wanted = []
    for p in (tp.h1.strip(), _clean_title(tp.title)):
        if p and p.lower() not in [w.lower() for w in wanted]:
            wanted.append(p)
    # distinctive phrases only count as anchors if they describe the page: they must share a word with its title or H1
    head = set(_TOKEN.findall((tp.title + " " + tp.h1).lower())) - _STOP
    phrases = wanted + [k for k in text_index.keyphrases(target, 12)
                        if k.lower() not in [w.lower() for w in wanted] and head & set(k.split())]
    body = (sp.body if sp is not None and sp.body else (sp.meta if sp is not None else "")) or ""
    if body:
        for ph in phrases:
            m = re.search(r"\b" + r"\s+".join(re.escape(w) for w in ph.split()) + r"\b", body, flags=re.I)
            if m:
                return dict(suggested_anchor=m.group(0), anchor_basis="Phrase already on the linking page",
                            context=_sentence_around(body, m.start(), m.end()))
    fallback = tp.h1.strip() or _clean_title(tp.title) or (phrases[0] if phrases else "")
    why = ("No natural phrase found on the linking page. Add one by hand" if body
           else "No body text for the linking page. Place by hand")
    return dict(suggested_anchor=fallback, anchor_basis=why, context="")


# --------------------------------------------------------------------------------------
# Suggestions
# --------------------------------------------------------------------------------------

SUGGEST_COLUMNS = ["action", "source", "destination", "similarity", "relevance", "suggested_anchor", "anchor_basis",
                   "context", "anchor_note", "source_pr_x_avg", "source_outlinks", "link_value", "priority", "method"]


def suggest_related_links(graph, scores: pd.DataFrame, targets: Iterable[str], index, *,
                          text_index: Optional[TextIndex] = None, anchors: Optional[Dict[str, Counter]] = None,
                          k: int = 5, min_percentile: float = 90.0, min_similarity: float = 0.0,
                          equity_weight: float = 0.5, same_segment: bool = False, segment_level: int = 1,
                          max_source_outlinks: Optional[int] = None) -> pd.DataFrame:
    """For each target, the most related pages that do not already link to it, with an anchor for each.

    A candidate must be an indexable page that is not the target, does not already link to it, and is among
    the top (100 - min_percentile)% most similar of the pages that could still add a link (and above min_similarity).
    Candidates are ordered by  priority = similarity x ((1 - w) + w x link_value_percentile)  where
    link_value = source pr_x_avg / (outlinks + 1) and w = equity_weight (0 = relevance only).
    Feed the result to ipr.apply_scenario (action = add) to measure the effect on link strength.
    Targets with no vector are listed in result.attrs["skipped"].
    """
    sc = scores[scores.indexable].copy()
    sc["link_value"] = sc.pr_x_avg / (sc.outlinks + 1)
    sc["value_pct"] = sc.link_value.rank(pct=True)
    if max_source_outlinks:
        sc = sc[sc.outlinks <= max_source_outlinks]
    anchors = anchors or {}
    idx, skipped, rows = graph.idx, [], []
    for t in targets:
        t = ipr.norm_url(t)
        if t not in idx or not index.has(t):
            skipped.append(t)
            continue
        sims = index.sim(t)
        simmap = dict(zip(index.urls, sims.tolist()))
        others = np.delete(sims, index.pos[t])
        existing = {graph.nodes[i] for i in graph.src[graph.dst == idx[t]].tolist()}
        cand = sc[(sc.url != t) & (~sc.url.isin(existing))].copy()
        cand["similarity"] = cand.url.map(simmap)
        cand = cand[cand.similarity.notna()]
        # the relevance gate is applied to pages that could still add a link, not to ones that already do
        pool = cand.similarity.to_numpy()
        cutoff = max(float(np.percentile(pool, min_percentile)) if len(pool) else 0.0, min_similarity)
        cand = cand[cand.similarity >= cutoff]
        if same_segment:
            seg = ipr.segment_of(t, None, segment_level)
            cand = cand[cand.url.map(lambda u: ipr.segment_of(u, None, segment_level)) == seg]
        cand["priority"] = cand.similarity * ((1 - equity_weight) + equity_weight * cand.value_pct)
        cand = cand.sort_values("priority", ascending=False).head(k)
        existing_anchors = anchors.get(t, Counter())
        total = sum(existing_anchors.values())
        for r in cand.itertuples():
            pct = float((others < r.similarity).mean() * 100)
            a = suggest_anchor(text_index, r.url, t)
            note = ""
            low = a["suggested_anchor"].lower()
            if low and total >= 3 and existing_anchors.get(low, 0) / total >= 0.3:
                note = f"{existing_anchors[low] / total:.0%} of the {total} links to this page already use this anchor. Vary it."
            rows.append({"action": "add", "source": r.url, "destination": t, "similarity": float(r.similarity),
                         "relevance": "Very close" if pct >= 99 else "Close" if pct >= 95 else "Related",
                         **a, "anchor_note": note, "source_pr_x_avg": r.pr_x_avg, "source_outlinks": int(r.outlinks),
                         "link_value": r.link_value, "priority": float(r.priority), "method": index.method})
    out = pd.DataFrame(rows, columns=SUGGEST_COLUMNS)
    out.attrs["skipped"] = skipped
    return out
