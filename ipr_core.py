"""Internal PageRank scoring and pre/post comparison from Screaming Frog exports.

The engine has no UI. ``ipr_cli.py`` and ``app.py`` (Streamlit) are thin wrappers.

Typical use::

    pre_g  = build_graph(read_sf_csv("pre_all_inlinks.csv"), read_sf_csv("pre_internal_all.csv"))
    post_g = build_graph(read_sf_csv("post_all_inlinks.csv"), read_sf_csv("post_internal_all.csv"))
    pre, post = score_graph(pre_g), score_graph(post_g)
    result = compare(pre, post, mapping=load_mapping(read_sf_csv("redirect_map.csv")))
    result["losers"]

"Internal PageRank" here is a model of how link equity flows through your own link graph. It is
not Google's PageRank. Use it to compare two states of the same site, not as an absolute number.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

# --------------------------------------------------------------------------------------
# Reading Screaming Frog exports
# --------------------------------------------------------------------------------------

_WANTED = {
    "type", "source", "destination", "follow", "link position", "status code",  # All Inlinks
    "address", "content type", "indexability", "redirect url", "crawl depth", "link score",  # Internal
}


def read_sf_csv(file_or_path, usecols_all: bool = False) -> pd.DataFrame:
    """Read a Screaming Frog CSV export, keeping only the columns the engine needs.

    Reading just the needed columns keeps big All Inlinks exports (millions of rows) light.
    """
    kwargs = dict(dtype=str, low_memory=False, encoding="utf-8-sig", keep_default_na=False)
    if not usecols_all:
        kwargs["usecols"] = lambda c: c.strip().lower() in _WANTED
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    df = pd.read_csv(file_or_path, **kwargs)
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


# --------------------------------------------------------------------------------------
# URL helpers
# --------------------------------------------------------------------------------------

def norm_url(u: str, ignore_host: bool = False, ignore_scheme: bool = False,
             trailing_slash: str = "keep") -> str:
    """Normalize a URL. Always lowercases scheme/host, drops fragments and default ports.

    ignore_host:    return only path?query (match staging to production by path)
    ignore_scheme:  treat http and https as the same
    trailing_slash: "keep" or "strip"
    """
    u = (u or "").strip()
    if not u or u.lower() == "nan":
        return ""
    p = urlsplit(u)
    scheme, host = p.scheme.lower(), re.sub(r":(80|443)$", "", p.netloc.lower())
    path = p.path or "/"
    if trailing_slash == "strip" and path != "/":
        path = path.rstrip("/") or "/"
    if ignore_host:
        return urlunsplit(("", "", path, p.query, ""))
    if ignore_scheme:
        scheme = ""
    return urlunsplit((scheme, host, path, p.query, ""))


def segment_of(url: str, segments: Optional[Sequence[Tuple[str, str]]] = None, level: int = 1) -> str:
    """Return the segment (template/directory) a URL belongs to."""
    path = urlsplit(url).path or "/"
    if segments:
        for name, rx in segments:
            if re.search(rx, path):
                return name
        return "(other)"
    parts = [p for p in path.split("/") if p]
    dirs = parts if path.endswith("/") else parts[:-1]
    return "/" + "/".join(dirs[:level]) if dirs else "/ (root level)"


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------

@dataclass
class LinkGraph:
    nodes: List[str]
    src: np.ndarray
    dst: np.ndarray
    w: np.ndarray
    meta: pd.DataFrame            # index = url: indexable(bool), status(float), sf_link_score(float), sf_depth(float)
    home: Optional[str]
    broken: pd.DataFrame          # internal links pointing at 4xx/5xx URLs
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def idx(self) -> Dict[str, int]:
        return {u: i for i, u in enumerate(self.nodes)}


def parse_position_weights(text: str) -> Dict[str, float]:
    """'content=1,navigation=0.5,footer=0.2,default=1' -> dict"""
    out: Dict[str, float] = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = float(v)
    return out


def build_graph(inlinks: pd.DataFrame, internal: Optional[pd.DataFrame] = None, *,
                follow_only: bool = True,
                include_positions: Optional[Iterable[str]] = None,
                exclude_positions: Optional[Iterable[str]] = None,
                position_weights: Optional[Dict[str, float]] = None,
                resolve_redirects: bool = True,
                home: Optional[str] = None) -> LinkGraph:
    """Build the internal link graph from Screaming Frog exports.

    inlinks   Bulk Export > Links > All Inlinks (needs Source, Destination; uses Type, Follow,
              Link Position, Status Code when present). All Outlinks works too.
    internal  Internal > All export (Address, Status Code, Content Type, Indexability,
              Redirect URL, Crawl Depth, Link Score). Optional but recommended.
    """
    sc, dc = _col(inlinks, "source", "from"), _col(inlinks, "destination", "to")
    if not sc or not dc:
        raise ValueError("The inlinks file needs 'Source' and 'Destination' columns. "
                         "Export Bulk Export > Links > All Inlinks from Screaming Frog.")
    n_raw = len(inlinks)
    clean = lambda s: s.astype(str).map(lambda u: norm_url(u))
    df = pd.DataFrame({"source": clean(inlinks[sc]), "destination": clean(inlinks[dc])})
    mask = (df.source != "") & (df.destination != "")

    # --- link type / redirect rows
    redirects: Dict[str, str] = {}
    tcol = _col(inlinks, "type")
    if tcol:
        types = inlinks[tcol].astype(str).str.lower()
        is_red = types.str.contains("redirect")
        for s, d in zip(df.source[is_red], df.destination[is_red]):
            redirects[s] = d
        mask &= types.str.contains("hyperlink")

    # --- follow / position filters
    n_before = int(mask.sum())
    fcol = _col(inlinks, "follow")
    if fcol and follow_only:
        mask &= ~inlinks[fcol].astype(str).str.lower().isin(["false", "0", "no"])
    n_nofollow_dropped = n_before - int(mask.sum())

    pcol = _col(inlinks, "link position")
    pos = inlinks[pcol].astype(str).str.lower() if pcol else pd.Series("", index=inlinks.index)
    if include_positions:
        mask &= pos.isin([p.lower() for p in include_positions])
    if exclude_positions:
        mask &= ~pos.isin([p.lower() for p in exclude_positions])

    if position_weights:
        wmap = {k.lower(): float(v) for k, v in position_weights.items()}
        dflt = wmap.get("default", 1.0)
        df["weight"] = pos.map(lambda p: wmap.get(p, dflt)).astype(float)
    else:
        df["weight"] = 1.0
    df = df[mask]

    # --- destination status without an Internal export
    status_map: Dict[str, float] = {}
    scol = _col(inlinks, "status code")
    if scol:
        st = pd.to_numeric(inlinks[scol], errors="coerce")
        status_map.update(dict(zip(df_all_dest(inlinks, dc), st)))

    # --- node universe + meta
    if internal is not None:
        ac = _col(internal, "address", "url")
        if not ac:
            raise ValueError("The Internal export needs an 'Address' column.")
        addr = internal[ac].astype(str).map(lambda u: norm_url(u))
        status = pd.to_numeric(internal[_col(internal, "status code")], errors="coerce") \
            if _col(internal, "status code") else pd.Series(200.0, index=internal.index)
        ctc = _col(internal, "content type")
        ctype = internal[ctc].astype(str).str.lower() if ctc else pd.Series("", index=internal.index)
        is_html = ctype.str.contains("html") | (ctype == "")
        ok = status.between(200, 299) | status.isna()
        icol = _col(internal, "indexability")
        indexable = internal[icol].astype(str).str.lower().eq("indexable") if icol \
            else pd.Series(True, index=internal.index)
        rcol = _col(internal, "redirect url")
        if rcol:
            is3 = status.between(300, 399)
            for a, r in zip(addr[is3], internal[rcol][is3].astype(str).map(lambda u: norm_url(u))):
                if r:
                    redirects[a] = r
        status_map.update(dict(zip(addr, status)))
        keep = is_html & ok & (addr != "")
        info = pd.DataFrame({"url": addr[keep], "indexable": indexable[keep].values,
                             "status": status[keep].values})
        dcol = _col(internal, "crawl depth")
        info["sf_depth"] = pd.to_numeric(internal[dcol][keep], errors="coerce").values if dcol else np.nan
        lcol = _col(internal, "link score")
        info["sf_link_score"] = pd.to_numeric(internal[lcol][keep], errors="coerce").values if lcol else np.nan
        info = info.drop_duplicates("url").set_index("url")
        node_set = set(info.index)
    else:
        urls = set(df.source) | set(df.destination)
        bad = {u for u in urls if status_map.get(u, 200) is not None and status_map.get(u, 200) >= 300}
        bad |= set(redirects)
        node_set = urls - bad
        info = pd.DataFrame({"indexable": True, "status": [status_map.get(u, 200.0) for u in sorted(node_set)],
                             "sf_depth": np.nan, "sf_link_score": np.nan}, index=sorted(node_set))

    # --- resolve redirects and split broken links
    def final(u: str) -> str:
        seen = 0
        while u in redirects and seen < 10:
            u, seen = redirects[u], seen + 1
        return u

    n_resolved = 0
    broken_rows = []
    if resolve_redirects and redirects:
        orig = df.destination.copy()
        df["destination"] = df.destination.map(final)
        n_resolved = int((orig != df.destination).sum())
    stat_of = lambda u: status_map.get(u, np.nan)
    is_broken = df.destination.map(lambda u: u not in node_set and stat_of(u) >= 400)
    if is_broken.any():
        b = df[is_broken][["source", "destination"]].copy()
        b["status"] = b.destination.map(stat_of)
        broken_rows = b
    broken = pd.DataFrame(broken_rows) if len(broken_rows) else pd.DataFrame(columns=["source", "destination", "status"])

    df = df[df.source.isin(node_set) & df.destination.isin(node_set) & (df.source != df.destination)]
    df = df.groupby(["source", "destination"], as_index=False)["weight"].max()

    nodes = sorted(node_set)
    idx = {u: i for i, u in enumerate(nodes)}
    src = df.source.map(idx).to_numpy(dtype=np.int64)
    dst = df.destination.map(idx).to_numpy(dtype=np.int64)
    w = df.weight.to_numpy(dtype=float)
    info = info.reindex(nodes)

    # --- home page
    home_url = norm_url(home) if home else None
    if home_url and home_url not in idx:
        home_url = None
    if not home_url:
        if info["sf_depth"].notna().any() and (info["sf_depth"] == 0).any():
            home_url = info.index[info["sf_depth"] == 0][0]
        else:
            roots = [u for u in nodes if (urlsplit(u).path in ("", "/")) and not urlsplit(u).query]
            if roots:
                home_url = sorted(roots, key=lambda u: (len(u), u))[0]
            elif nodes:
                inl = np.bincount(dst, minlength=len(nodes))
                home_url = nodes[int(inl.argmax())]

    stats = {"link_rows_in_export": n_raw, "edges_used": int(len(df)), "nodes": len(nodes),
             "nofollow_rows_dropped": n_nofollow_dropped, "links_via_redirects_resolved": n_resolved,
             "broken_link_rows": int(len(broken))}
    return LinkGraph(nodes, src, dst, w, info, home_url, broken, stats)


def df_all_dest(inlinks: pd.DataFrame, dc: str):
    return inlinks[dc].astype(str).map(lambda u: norm_url(u)).tolist()


# --------------------------------------------------------------------------------------
# PageRank and scoring
# --------------------------------------------------------------------------------------

def pagerank(graph: LinkGraph, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """Weighted PageRank by power iteration. Dangling nodes redistribute uniformly. Sums to 1."""
    n = len(graph.nodes)
    if n == 0:
        return np.zeros(0)
    out_w = np.bincount(graph.src, weights=graph.w, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = graph.w / out_w[graph.src]
    A = sp.csr_matrix((vals, (graph.dst, graph.src)), shape=(n, n))
    dangling = out_w == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        nr = damping * (A @ r + r[dangling].sum() / n) + (1 - damping) / n
        done = np.abs(nr - r).sum() < tol
        r = nr
        if done:
            break
    return r / r.sum()


def click_depth(graph: LinkGraph) -> np.ndarray:
    """Clicks from the home page along followed internal links. NaN when unreachable."""
    n = len(graph.nodes)
    depth = np.full(n, np.nan)
    if not graph.home or n == 0:
        return depth
    h = graph.idx[graph.home]
    A = sp.csr_matrix((np.ones(len(graph.src)), (graph.src, graph.dst)), shape=(n, n))
    d = dijkstra(A, directed=True, indices=h, unweighted=True)
    d[np.isinf(d)] = np.nan
    return d


def score_graph(graph: LinkGraph, damping: float = 0.85) -> pd.DataFrame:
    """One row per page: PageRank, readable scores, link counts, depth, orphan flag.

    unreachable True when no chain of followed links leads to the page from the home page
    pr          true PageRank. Always between 0 and 1 and sums to 1 across the site, so each
                page's value is a tiny share that shrinks as the site gets bigger.
    pr_x_avg    PageRank as a MULTIPLE of the average page (pr x number of pages).
                1.0 = average, 3.0 = three times average, 0.2 = a fifth of average. Values above
                1.0 are normal (they are not PageRank itself). Use this to compare crawls.
    score       0-100 log scale of pr (similar idea to Screaming Frog's Link Score)
    percentile  0-100 rank among pages in this crawl
    """
    n = len(graph.nodes)
    pr = pagerank(graph, damping)
    inl = np.bincount(graph.dst, minlength=n)
    outl = np.bincount(graph.src, minlength=n)
    lo, hi = (np.log(pr.min()), np.log(pr.max())) if n else (0, 0)
    score = 100 * (np.log(pr) - lo) / (hi - lo) if hi > lo else np.full(n, 50.0)
    df = pd.DataFrame({"url": graph.nodes, "pr": pr, "pr_x_avg": pr * n, "score": score,
                       "percentile": pd.Series(pr).rank(pct=True).to_numpy() * 100,
                       "inlinks": inl, "outlinks": outl, "depth": click_depth(graph)})
    df["orphan"] = (df.inlinks == 0) & (df.url != graph.home)
    # reachable = a followed-link path exists from the home page; "unreachable" catches whole
    # sections that only link to themselves, which a plain inlink count would miss
    df["unreachable"] = df.depth.isna() & (df.url != graph.home)
    meta = graph.meta.reset_index(drop=True)
    df["indexable"] = meta["indexable"].fillna(True).astype(bool).to_numpy()
    df["status"] = meta["status"].to_numpy()
    df["sf_link_score"] = meta["sf_link_score"].to_numpy()
    return df.sort_values("pr", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Pre / post comparison
# --------------------------------------------------------------------------------------

def load_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Redirect map CSV -> {old_url: new_url}. Accepts Source/Target, Old/New or From/To columns."""
    s = _col(df, "source", "old", "old url", "from", "old_url")
    t = _col(df, "target", "new", "new url", "to", "new_url", "destination")
    if not s or not t:
        raise ValueError("The redirect map needs Source and Target columns (or Old/New, From/To).")
    out = {}
    for a, b in zip(df[s].astype(str), df[t].astype(str)):
        a, b = a.strip(), b.strip()
        if a and b and a.lower() != "nan" and b.lower() != "nan":
            out[a] = b
    return out


def compare(pre: pd.DataFrame, post: pd.DataFrame, mapping: Optional[Dict[str, str]] = None, *,
            ignore_host: bool = False, trailing_slash: str = "keep", indexable_only: bool = True,
            loser_ratio: float = 0.5, pct_drop: float = 20.0, depth_increase: int = 2,
            top_k: int = 100, segments: Optional[Sequence[Tuple[str, str]]] = None,
            segment_level: int = 1) -> Dict[str, object]:
    """Compare two scored crawls, remapping old URLs to new ones through the redirect map.

    When several old URLs map to one new URL their pr_x_avg values are summed, so a consolidation
    is compared with the combined equity it should inherit. Old URLs missing from the map fall
    back to the identical URL (or the identical path with ignore_host) in the post crawl.
    """
    kf = lambda u: norm_url(u, ignore_host=ignore_host, trailing_slash=trailing_slash)
    post_key = {}
    for u in post.url:
        post_key.setdefault(kf(u), u)
    mapmap = {kf(k): kf(v) for k, v in (mapping or {}).items()}

    use = pre[pre.indexable] if indexable_only else pre
    use = use.copy()
    use["key"] = use.url.map(kf)
    use["via_map"] = use.key.isin(mapmap)
    use["target_key"] = use.key.map(lambda k: mapmap.get(k, k))
    use["target"] = use.target_key.map(post_key)
    matched = use[use.target.notna()]
    lost = use[use.target.isna()].copy()
    lost["reason"] = np.where(lost.via_map, "map target not found in post crawl",
                              "no map entry and no matching URL in post crawl")
    lost["intended_target"] = np.where(lost.via_map, lost.target_key, "")
    unmapped_old = lost[["url", "intended_target", "reason", "pr_x_avg", "percentile", "inlinks", "depth"]] \
        .rename(columns={"url": "old_url", "pr_x_avg": "pre_pr_x_avg", "percentile": "pre_percentile",
                         "inlinks": "pre_inlinks", "depth": "pre_depth"}) \
        .sort_values("pre_pr_x_avg", ascending=False).reset_index(drop=True)

    m = matched.sort_values("pr_x_avg", ascending=False)
    pairs = m.groupby("target", sort=False).agg(
        old_urls=("url", lambda s: " | ".join(list(s)[:3]) + (f" (+{len(s) - 3} more)" if len(s) > 3 else "")),
        old_count=("url", "size"),
        pre_pr_x_avg=("pr_x_avg", "sum"),
        pre_pr=("pr", "sum"),
        pre_percentile=("percentile", "first"),
        pre_inlinks=("inlinks", "first"),
        pre_depth=("depth", "first"),
    ).reset_index().rename(columns={"target": "new_url"})
    pcols = post.set_index("url")[["pr", "pr_x_avg", "percentile", "inlinks", "depth", "orphan", "unreachable"]] \
        .rename(columns={"pr": "post_pr", "pr_x_avg": "post_pr_x_avg", "percentile": "post_percentile",
                         "inlinks": "post_inlinks", "depth": "post_depth", "orphan": "post_orphan",
                         "unreachable": "post_unreachable"})
    pairs = pairs.merge(pcols, left_on="new_url", right_index=True, how="left")
    pairs["delta"] = pairs.post_pr_x_avg - pairs.pre_pr_x_avg
    pairs["ratio"] = pairs.post_pr_x_avg / pairs.pre_pr_x_avg
    pairs["percentile_change"] = pairs.post_percentile - pairs.pre_percentile
    pairs["depth_change"] = pairs.post_depth - pairs.pre_depth

    def flags(r) -> str:
        f = []
        if r.ratio < loser_ratio or r.percentile_change <= -pct_drop:
            f.append("LOSER")
        if r.ratio > 1 / loser_ratio:
            f.append("GAINER")
        if pd.notna(r.depth_change) and r.depth_change >= depth_increase:
            f.append("DEEPER")
        if r.post_inlinks == 0:
            f.append("NO_INLINKS")
        elif r.post_unreachable and pd.notna(r.pre_depth):
            f.append("UNREACHABLE")
        return ", ".join(f)

    pairs["flags"] = pairs.apply(flags, axis=1) if len(pairs) else ""
    pairs = pairs.sort_values("pre_pr_x_avg", ascending=False).reset_index(drop=True)

    losers = pairs[pairs["flags"].str.contains("LOSER")].sort_values("delta").reset_index(drop=True)
    gainers = pairs[pairs["flags"].str.contains("GAINER")].sort_values("delta", ascending=False).reset_index(drop=True)
    deeper = pairs[pairs["flags"].str.contains("DEEPER|UNREACHABLE")].sort_values("pre_pr_x_avg", ascending=False).reset_index(drop=True)

    targets = set(pairs.new_url)
    pidx = post[post.indexable] if "indexable" in post else post
    new_pages = pidx[~pidx.url.isin(targets)][["url", "pr_x_avg", "percentile", "inlinks", "depth", "orphan"]] \
        .rename(columns={"url": "new_url", "pr_x_avg": "post_pr_x_avg", "percentile": "post_percentile",
                         "inlinks": "post_inlinks", "depth": "post_depth"}) \
        .sort_values("post_pr_x_avg", ascending=False).reset_index(drop=True)
    bad = pidx[pidx.orphan | pidx.unreachable]
    orphans_post = bad[["url", "pr_x_avg", "inlinks", "depth"]].assign(
        reason=np.where(bad.orphan, "no internal links in", "not reachable from the home page"),
        is_mapped_target=bad.url.isin(targets)) \
        .rename(columns={"url": "new_url", "pr_x_avg": "post_pr_x_avg", "inlinks": "post_inlinks", "depth": "post_depth"}) \
        .sort_values("post_pr_x_avg", ascending=False).reset_index(drop=True)

    kept = np.nan
    if len(use):
        top_pre = m.nlargest(min(top_k, len(m)), "pr_x_avg").target
        top_post = set(pidx.nlargest(min(top_k, len(pidx)), "pr_x_avg").url)
        kept = float(np.mean([t in top_post for t in top_pre])) if len(top_pre) else np.nan
    corr = pairs[["pre_pr_x_avg", "post_pr_x_avg"]].corr(method="spearman").iloc[0, 1] if len(pairs) > 2 else np.nan
    total_pre = float(use.pr_x_avg.sum())
    summary = {
        "pages_pre": int(len(pre)), "pages_post": int(len(post)),
        "old_urls_compared": int(len(use)), "matched_to_post_crawl": int(len(matched)),
        "old_urls_with_no_destination": int(len(lost)),
        "share_of_pre_equity_with_no_destination": float(lost.pr_x_avg.sum() / total_pre) if total_pre else 0.0,
        "losers": int(len(losers)), "gainers": int(len(gainers)), "deeper_pages": int(len(deeper)),
        "median_ratio": float(pairs.ratio.median()) if len(pairs) else np.nan,
        f"top_{top_k}_still_in_top_{top_k}": kept,
        "rank_stability_spearman": float(corr) if pd.notna(corr) else np.nan,
        "post_orphans_or_unreachable": int(len(orphans_post)),
    }

    seg_pre = segment_rollup(pre if not indexable_only else pre[pre.indexable], segments, segment_level)
    seg_post = segment_rollup(post if not indexable_only else post[post.indexable], segments, segment_level)
    return {"summary": summary, "pairs": pairs, "losers": losers, "gainers": gainers, "deeper": deeper,
            "unmapped_old": unmapped_old, "new_pages": new_pages, "orphans_post": orphans_post,
            "segments": segment_compare(seg_pre, seg_post)}


# --------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------

def parse_segments(text: str) -> List[Tuple[str, str]]:
    """'Products=^/products/, Blog=^/blog/' one per line or comma separated -> [(name, regex)]"""
    out = []
    for line in re.split(r"[\n]+", text or ""):
        if "=" in line:
            n, rx = line.split("=", 1)
            out.append((n.strip(), rx.strip()))
    return out


def segment_rollup(scores: pd.DataFrame, segments=None, level: int = 1) -> pd.DataFrame:
    """PageRank share and averages per segment (first directory by default, or your regexes)."""
    if scores.empty:
        return pd.DataFrame(columns=["segment", "pages", "pr_share_pct", "avg_pr_x_avg", "avg_score",
                                     "avg_inlinks", "median_depth"])
    d = scores.copy()
    d["segment"] = d.url.map(lambda u: segment_of(u, segments, level))
    total = d.pr.sum()
    g = d.groupby("segment").agg(pages=("url", "size"), pr_sum=("pr", "sum"), avg_pr_x_avg=("pr_x_avg", "mean"),
                                 avg_score=("score", "mean"), avg_inlinks=("inlinks", "mean"),
                                 median_depth=("depth", "median")).reset_index()
    g["pr_share_pct"] = g.pr_sum / total * 100
    return g.drop(columns="pr_sum")[["segment", "pages", "pr_share_pct", "avg_pr_x_avg", "avg_score",
                                     "avg_inlinks", "median_depth"]].sort_values("pr_share_pct", ascending=False) \
        .reset_index(drop=True)


def segment_compare(pre_roll: pd.DataFrame, post_roll: pd.DataFrame) -> pd.DataFrame:
    """Side by side segment tables. Segment names must match to line up (use your own regexes
    when directories are renamed in the migration)."""
    m = pre_roll.merge(post_roll, on="segment", how="outer", suffixes=("_pre", "_post"))
    m["share_change_pts"] = m.pr_share_pct_post.fillna(0) - m.pr_share_pct_pre.fillna(0)
    return m.sort_values("share_change_pts").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Scenarios: test an internal linking plan
# --------------------------------------------------------------------------------------

def apply_scenario(graph: LinkGraph, changes: pd.DataFrame) -> LinkGraph:
    """Apply proposed link changes and return a new graph.

    changes columns: action (add/remove), source, destination, optional weight.
    URLs not already in the graph are added as new pages (useful for staging crawls).
    """
    ac, sc, dc = _col(changes, "action"), _col(changes, "source"), _col(changes, "destination")
    wc = _col(changes, "weight")
    if not (ac and sc and dc):
        raise ValueError("Scenario file needs columns: action (add/remove), source, destination.")
    ch = pd.DataFrame({"action": changes[ac].astype(str).str.lower().str.strip(),
                       "source": changes[sc].astype(str).map(lambda u: norm_url(u)),
                       "destination": changes[dc].astype(str).map(lambda u: norm_url(u)),
                       "weight": pd.to_numeric(changes[wc], errors="coerce").fillna(1.0) if wc else 1.0})
    nodes = list(graph.nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    for u in pd.concat([ch.loc[ch.action == "add", "source"], ch.loc[ch.action == "add", "destination"]]):
        if u and u not in idx:
            idx[u] = len(nodes)
            nodes.append(u)
    meta = graph.meta.reindex(nodes)
    meta["indexable"] = meta["indexable"].fillna(True)
    meta["status"] = meta["status"].fillna(200.0)

    edges = pd.DataFrame({"s": graph.src, "d": graph.dst, "w": graph.w})
    rm = ch[ch.action == "remove"]
    if len(rm):
        rm_pairs = pd.DataFrame({"s": rm.source.map(idx), "d": rm.destination.map(idx)}).dropna().astype(np.int64)
        edges = edges.merge(rm_pairs.assign(_x=1), on=["s", "d"], how="left")
        edges = edges[edges._x.isna()].drop(columns="_x")
    add = ch[ch.action == "add"]
    if len(add):
        add_e = pd.DataFrame({"s": add.source.map(idx), "d": add.destination.map(idx), "w": add.weight.values})
        add_e = add_e[add_e.s != add_e.d]
        edges = pd.concat([edges, add_e], ignore_index=True)
    edges = edges.groupby(["s", "d"], as_index=False)["w"].max()
    stats = dict(graph.stats, edges_used=int(len(edges)), nodes=len(nodes))
    return replace(graph, nodes=nodes, src=edges.s.to_numpy(np.int64), dst=edges.d.to_numpy(np.int64),
                   w=edges.w.to_numpy(float), meta=meta, stats=stats)


def scenario_report(base: pd.DataFrame, scen: pd.DataFrame, targets: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Before/after PageRank for a linking scenario."""
    b = base.set_index("url")[["pr_x_avg", "percentile", "inlinks", "indexable"]]
    s = scen.set_index("url")[["pr_x_avg", "percentile", "inlinks"]]
    m = b.join(s, lsuffix="_before", rsuffix="_after", how="outer")
    m["pr_x_avg_before"] = m.pr_x_avg_before.fillna(0.0)
    m["delta"] = m.pr_x_avg_after - m.pr_x_avg_before
    m["change_pct"] = np.where(m.pr_x_avg_before > 0, m.delta / m.pr_x_avg_before * 100, np.nan)
    m = m.reset_index().rename(columns={"index": "url"})
    out = {"all": m.sort_values("delta", ascending=False).reset_index(drop=True)}
    out["gainers"] = m[m.delta > 1e-9].sort_values("delta", ascending=False).head(50).reset_index(drop=True)
    out["losers"] = m[m.delta < -1e-9].sort_values("delta").head(50).reset_index(drop=True)
    tset = {norm_url(t) for t in (targets or [])}
    if tset:
        tt = m[m.url.isin(tset)]
        out["targets"] = tt.sort_values("delta", ascending=False).reset_index(drop=True)
        out["summary"] = {"targets": int(len(tt)),
                          "target_pr_x_avg_gain": float(tt.delta.sum()),
                          "others_pr_x_avg_change": float(m[~m.url.isin(tset)].delta.sum()),
                          "avg_target_pct_change": float(tt["change_pct"].mean()) if len(tt) else np.nan}
    return out


def suggest_links(graph: LinkGraph, scores: pd.DataFrame, targets: Iterable[str], k: int = 5,
                  same_segment: bool = False, segment_level: int = 1,
                  min_source_pr_x_avg: float = 0.0) -> pd.DataFrame:
    """For each target, suggest the strongest pages that do not already link to it.

    Ranking proxy: link_value = source pr_x_avg / (outlinks + 1). A page with huge equity but 300
    outgoing links passes little per link, so the proxy favors strong pages with few links.
    Feed the result to apply_scenario (action=add) to measure the combined effect.
    """
    idx = graph.idx
    inl_by_target: Dict[int, set] = {}
    sc = scores[scores.indexable & (scores.pr_x_avg >= min_source_pr_x_avg)].copy()
    sc["link_value"] = sc.pr_x_avg / (sc.outlinks + 1)
    sc = sc.sort_values("link_value", ascending=False)
    rows = []
    for t in targets:
        t = norm_url(t)
        if t not in idx:
            continue
        ti = idx[t]
        existing = set(graph.src[graph.dst == ti].tolist())
        existing_urls = {graph.nodes[i] for i in existing}
        cand = sc[(sc.url != t) & (~sc.url.isin(existing_urls))]
        if same_segment:
            seg = segment_of(t, None, segment_level)
            cand = cand[cand.url.map(lambda u: segment_of(u, None, segment_level)) == seg]
        for _, r in cand.head(k).iterrows():
            rows.append({"action": "add", "source": r.url, "destination": t,
                         "source_pr_x_avg": r.pr_x_avg, "source_outlinks": int(r.outlinks),
                         "link_value": r.link_value})
    return pd.DataFrame(rows, columns=["action", "source", "destination", "source_pr_x_avg",
                                       "source_outlinks", "link_value"])


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------

def reports_to_zip(reports: Dict[str, pd.DataFrame]) -> bytes:
    """Bundle DataFrames as CSVs inside a zip (for the download button)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in reports.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
    return buf.getvalue()


def summary_frame(summary: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(summary.keys()), "value": list(summary.values())})
