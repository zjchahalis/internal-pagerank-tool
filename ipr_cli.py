#!/usr/bin/env python3
"""Command line wrapper for the internal PageRank engine.

  Score one crawl
    python ipr_cli.py score --inlinks all_inlinks.csv --internal internal_all.csv --out out/

  Compare before and after a migration
    python ipr_cli.py compare \
        --pre-inlinks pre_all_inlinks.csv   --pre-internal pre_internal_all.csv \
        --post-inlinks post_all_inlinks.csv --post-internal post_internal_all.csv \
        --map redirect_map.csv --out out/

  Test a linking plan (your own changes file, or let the tool suggest links to target pages)
    python ipr_cli.py scenario --inlinks all_inlinks.csv --internal internal_all.csv \
        --changes plan.csv --out out/
    python ipr_cli.py scenario --inlinks all_inlinks.csv --internal internal_all.csv \
        --suggest-for money_pages.txt --k 5 --out out/
    python ipr_cli.py scenario --inlinks all_inlinks.csv --internal internal_all.csv \
        --suggest-for money_pages.txt --embeddings embeddings.csv --page-text page_text.csv --out out/
    (with --embeddings, suggestions are the most RELATED pages; without it, TF-IDF over titles, H1s, meta
     descriptions and any --page-text is used. Add --strength-only for the old strength-only suggestions.)

Run ``python ipr_cli.py <command> -h`` for every option.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import ipr_core as ipr
import ipr_related as rel


def _graph_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor (default 0.85)")
    p.add_argument("--include-nofollow", action="store_true", help="count nofollow links as normal links")
    p.add_argument("--exclude-positions", default="", help="comma list of Link Position values to ignore, e.g. footer")
    p.add_argument("--position-weights", default="",
                   help="weight links by position, e.g. 'content=1,navigation=0.5,footer=0.2,default=1'")
    p.add_argument("--no-resolve-redirects", action="store_true", help="do not pass equity through redirects")
    p.add_argument("--home", default=None, help="home page URL (auto-detected by default)")


def _build(inlinks: str, internal: str | None, a: argparse.Namespace) -> ipr.LinkGraph:
    return ipr.build_graph(
        ipr.read_sf_csv(inlinks), ipr.read_sf_csv(internal) if internal else None,
        follow_only=not a.include_nofollow,
        exclude_positions=[x.strip() for x in a.exclude_positions.split(",") if x.strip()] or None,
        position_weights=ipr.parse_position_weights(a.position_weights) or None,
        resolve_redirects=not a.no_resolve_redirects, home=a.home)


def _save(out: str, reports: dict) -> None:
    os.makedirs(out, exist_ok=True)
    for name, df in reports.items():
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        print(f"  wrote {os.path.join(out, name)}.csv  ({len(df)} rows)")


def _segments(a) -> list | None:
    if getattr(a, "segments", None):
        return ipr.parse_segments(open(a.segments, encoding="utf-8").read())
    return None


def cmd_score(a: argparse.Namespace) -> None:
    g = _build(a.inlinks, a.internal, a)
    s = ipr.score_graph(g, a.damping)
    print("Graph:", g.stats, "| home:", g.home)
    reports = {"scores": s,
               "segments": ipr.segment_rollup(s[s.indexable], _segments(a), a.segment_level),
               "orphans_unreachable": s[(s.orphan | s.unreachable) & s.indexable],
               "broken_links": g.broken}
    _save(a.out, reports)
    print("\nTop 10 pages by internal PageRank (pr_x_avg 1.0 = average page):")
    print(s[["url", "pr_x_avg", "score", "inlinks", "depth"]].head(10).to_string(index=False))


def cmd_compare(a: argparse.Namespace) -> None:
    pre_g = _build(a.pre_inlinks, a.pre_internal, a)
    post_g = _build(a.post_inlinks, a.post_internal, a)
    pre, post = ipr.score_graph(pre_g, a.damping), ipr.score_graph(post_g, a.damping)
    mapping = ipr.load_mapping(ipr.read_sf_csv(a.map, usecols_all=True)) if a.map else None
    res = ipr.compare(pre, post, mapping, ignore_host=a.ignore_host,
                      trailing_slash="strip" if a.strip_trailing_slash else "keep",
                      loser_ratio=a.loser_ratio, pct_drop=a.pct_drop, depth_increase=a.depth_increase,
                      segments=_segments(a), segment_level=a.segment_level)
    print("Pre graph: ", pre_g.stats, "\nPost graph:", post_g.stats)
    print("\nSummary")
    for k, v in res["summary"].items():
        print(f"  {k}: {round(v, 3) if isinstance(v, float) else v}")
    reports = {k: v for k, v in res.items() if k != "summary"}
    reports["summary"] = ipr.summary_frame(res["summary"])
    reports["pre_scores"], reports["post_scores"] = pre, post
    reports["post_broken_links"] = post_g.broken
    _save(a.out, reports)
    print("\nBiggest losers (absolute equity lost):")
    cols = ["new_url", "pre_pr_x_avg", "post_pr_x_avg", "ratio", "flags"]
    print(res["losers"][cols].head(10).to_string(index=False))


def cmd_scenario(a: argparse.Namespace) -> None:
    g = _build(a.inlinks, a.internal, a)
    base = ipr.score_graph(g, a.damping)
    if a.changes:
        changes = ipr.read_sf_csv(a.changes, usecols_all=True)
        targets = changes.loc[changes[ipr._col(changes, "action")].astype(str).str.lower() == "add",
                              ipr._col(changes, "destination")].tolist()
    elif a.suggest_for:
        targets = [l.strip() for l in open(a.suggest_for, encoding="utf-8") if l.strip()]
        if a.strength_only:
            changes = ipr.suggest_links(g, base, targets, k=a.k, same_segment=a.same_segment)
        else:
            internal = ipr.read_sf_csv(a.internal, usecols_all=True) if a.internal else None
            text = ipr.read_sf_csv(a.page_text, usecols_all=True) if a.page_text else None
            pages = rel.build_page_text(internal, text)
            text_index = rel.build_text_index(pages) if len(pages) else None
            if a.embeddings:
                index = rel.load_embeddings(rel.read_embeddings_csv(a.embeddings))
                print(f"Loaded {len(index.urls)} embeddings ({index.dims} dimensions); {index.dropped} rows skipped")
            elif text_index is not None:
                index = text_index
            else:
                sys.exit("Related-link suggestions need --embeddings, or text: --internal with titles/H1s or --page-text. "
                         "Use --strength-only for strength-only suggestions.")
            anchors = rel.anchor_counts(rel.read_anchor_table(a.inlinks))
            changes = rel.suggest_related_links(
                g, base, targets, index, text_index=text_index, anchors=anchors, k=a.k,
                min_percentile=a.min_percentile, min_similarity=a.min_similarity, equity_weight=a.equity_weight,
                same_segment=a.same_segment, max_source_outlinks=a.max_source_outlinks)
            if changes.attrs.get("skipped"):
                print("No vector or text for these targets, skipped:", ", ".join(changes.attrs["skipped"]))
            print(f"Method: {index.method}")
        print(f"Suggested {len(changes)} links for {len(targets)} target pages")
    else:
        sys.exit("Give --changes plan.csv or --suggest-for targets.txt")
    scen = ipr.score_graph(ipr.apply_scenario(g, changes), a.damping)
    rep = ipr.scenario_report(base, scen, targets)
    reports = {"scenario_changes": changes, "scenario_all": rep["all"],
               "scenario_gainers": rep["gainers"], "scenario_losers": rep["losers"]}
    if "targets" in rep:
        reports["scenario_targets"] = rep["targets"]
        print("Summary:", rep["summary"])
    _save(a.out, reports)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one crawl")
    s.add_argument("--inlinks", required=True)
    s.add_argument("--internal")
    s.add_argument("--segments", help="text file: Name=regex per line")
    s.add_argument("--segment-level", type=int, default=1)
    s.add_argument("--out", default="ipr_out")
    _graph_opts(s)
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare", help="compare pre and post crawls")
    c.add_argument("--pre-inlinks", required=True)
    c.add_argument("--pre-internal")
    c.add_argument("--post-inlinks", required=True)
    c.add_argument("--post-internal")
    c.add_argument("--map", help="redirect map CSV with Source,Target (old to new)")
    c.add_argument("--ignore-host", action="store_true", help="match URLs by path only (staging vs production)")
    c.add_argument("--strip-trailing-slash", action="store_true")
    c.add_argument("--loser-ratio", type=float, default=0.5, help="flag pages whose pr_x_avg falls below this fraction")
    c.add_argument("--pct-drop", type=float, default=20.0, help="or whose percentile falls by this many points")
    c.add_argument("--depth-increase", type=int, default=2)
    c.add_argument("--segments", help="text file: Name=regex per line")
    c.add_argument("--segment-level", type=int, default=1)
    c.add_argument("--out", default="ipr_out")
    _graph_opts(c)
    c.set_defaults(fn=cmd_compare)

    n = sub.add_parser("scenario", help="test a proposed internal linking plan")
    n.add_argument("--inlinks", required=True)
    n.add_argument("--internal")
    n.add_argument("--changes", help="CSV: action(add/remove),source,destination[,weight]")
    n.add_argument("--suggest-for", help="text file of target URLs, one per line")
    n.add_argument("--k", type=int, default=5, help="suggested links per target")
    n.add_argument("--embeddings", help="CSV: Address plus one numeric column per dimension (or one column holding the vector)")
    n.add_argument("--page-text", help="CSV: Address plus a body text column (for anchors and TF-IDF)")
    n.add_argument("--strength-only", action="store_true", help="old behaviour: rank by link strength only, ignore relevance")
    n.add_argument("--min-percentile", type=float, default=90.0, help="only the most similar (100 - N)%% of candidates are eligible")
    n.add_argument("--min-similarity", type=float, default=0.0, help="also require at least this cosine similarity")
    n.add_argument("--equity-weight", type=float, default=0.5, help="0 = order by relevance only, 1 = lean fully on link strength")
    n.add_argument("--max-source-outlinks", type=int, help="skip linking pages that already have more links than this")
    n.add_argument("--same-segment", action="store_true", help="only suggest sources in the target's directory")
    n.add_argument("--out", default="ipr_out")
    _graph_opts(n)
    n.set_defaults(fn=cmd_scenario)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
