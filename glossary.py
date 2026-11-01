"""Plain-English names and hover tooltips for every column and metric the app shows.

The engine (ipr_core.py) and the command line keep their technical column names. The app
translates them through this file, so the words people see can change without touching the maths.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

import pandas as pd

# column -> (label shown in the app, plain-English tooltip)
COLUMNS: Dict[str, Tuple[str, str]] = {
    # ---- pages
    "url": ("Page", "The page's web address."),
    "new_url": ("New page", "The page's address on the new (after) site."),
    "old_url": ("Old page", "The page's address on the old (before) site."),
    "old_urls": ("Old page(s)", "The old page or pages that now point to this new page. When several old pages were merged into one, up to three are shown."),
    "old_count": ("Old pages merged", "How many old pages were merged into this one new page. 1 means a simple one-to-one move."),
    "intended_target": ("Planned new page", "Where your redirect map says this old page should go."),
    "reason": ("Why", "Why this page is on the list."),
    "destination": ("Linked-to page", "The page being linked to."),
    "source": ("Linking page", "The page that holds the link."),
    "action": ("Action", "Add a link, or remove one."),
    "weight": ("Link weight", "How much of the linking page's strength this link passes. 1 is a normal link."),
    # ---- strength scores
    "pr_x_avg": ("Link strength (× average)", "How much link strength this page receives compared with an average page on the site. 1.0 = average, 3.0 = three times average, 0.2 = a fifth of average. Values above 1.0 are normal: strong pages like the home page score well above it."),
    "pr": ("PageRank (0 to 1)", "The page's true PageRank: its share of all the link strength on the site. Always between 0 and 1, and every page adds up to exactly 1. It gets tiny on big sites, which is why the 'Link strength' column is easier to read."),
    "score": ("Link score (0 to 100)", "A 0 to 100 score for how well linked the page is within this site. 100 is the best linked page. It is a log scale, so each step up means a lot more strength."),
    "percentile": ("Rank (percentile)", "How this page compares with the others on the site. 90 means it is better linked than 90% of pages."),
    "sf_link_score": ("Screaming Frog Link Score", "Screaming Frog's own link score for this page, if it was in your export. A useful second opinion on the numbers here."),
    # ---- links and position
    "inlinks": ("Links in", "How many pages link to this page."),
    "outlinks": ("Links out", "How many pages this page links to."),
    "depth": ("Clicks from home", "The fewest clicks it takes to reach this page from the home page. Lower is better. Blank means it can't be reached."),
    "orphan": ("No links in", "True when no other page links to this one."),
    "unreachable": ("Cut off from home", "True when you can't get to this page from the home page by clicking links, even if other pages link to it."),
    "indexable": ("Can be indexed", "True when search engines are allowed to index the page."),
    "status": ("Status code", "The server's response for this page. 200 means OK."),
    # ---- before / after
    "pre_pr_x_avg": ("Strength before (× avg)", "Link strength on the old site, as a multiple of the average page. When several old pages were merged, their strengths are added together."),
    "post_pr_x_avg": ("Strength after (× avg)", "Link strength on the new site, as a multiple of the average page."),
    "ratio": ("Kept (after ÷ before)", "How much of its link strength the page kept. 1.0 = unchanged, 0.5 = lost half, 2.0 = doubled."),
    "delta": ("Change in strength", "Strength after minus strength before, in 'times the average page'. Negative means the page lost strength."),
    "pre_pr": ("PageRank before (0 to 1)", "The page's true PageRank on the old site."),
    "post_pr": ("PageRank after (0 to 1)", "The page's true PageRank on the new site."),
    "pre_percentile": ("Rank before", "How the page ranked among the old site's pages (percentile). 90 = better linked than 90% of pages."),
    "post_percentile": ("Rank after", "How the page ranks among the new site's pages (percentile)."),
    "percentile_change": ("Rank change (points)", "Rank after minus rank before. A big negative number means the page fell down the site's pecking order."),
    "pre_inlinks": ("Links in before", "How many pages linked to it on the old site."),
    "post_inlinks": ("Links in after", "How many pages link to it on the new site."),
    "pre_depth": ("Clicks from home before", "Clicks from the home page on the old site."),
    "post_depth": ("Clicks from home after", "Clicks from the home page on the new site. Blank means it can't be reached."),
    "depth_change": ("Change in clicks from home", "Positive means the page is buried deeper than before."),
    "post_orphan": ("No links in (after)", "True when no page links to it on the new site."),
    "post_unreachable": ("Cut off from home (after)", "True when it can't be reached from the home page on the new site."),
    "flags": ("What changed", "Plain labels for what went wrong or right with this page. See the Glossary tab for each label."),
    "pre_pr_x_avg_sum": ("Strength before (× avg)", "Link strength on the old site."),
    "is_mapped_target": ("Was an old page's destination", "True when your redirect map sends an old page here, so a missing link in is a real problem."),
    # ---- sections
    "segment": ("Section", "A group of pages, by folder in the web address (for example /blog) or by the patterns you defined."),
    "pages": ("Pages", "How many pages are in this section."),
    "pages_pre": ("Pages before", "How many pages the section had on the old site."),
    "pages_post": ("Pages after", "How many pages the section has on the new site."),
    "pr_share_pct": ("Share of link strength (%)", "The part of the whole site's link strength that sits in this section."),
    "pr_share_pct_pre": ("Share before (%)", "The share of the old site's link strength held by this section."),
    "pr_share_pct_post": ("Share after (%)", "The share of the new site's link strength held by this section."),
    "avg_pr_x_avg": ("Average strength (× avg)", "The average link strength of pages in this section, as a multiple of the average page."),
    "avg_pr_x_avg_pre": ("Average strength before (× avg)", "Average link strength per page in this section on the old site."),
    "avg_pr_x_avg_post": ("Average strength after (× avg)", "Average link strength per page in this section on the new site."),
    "avg_score": ("Average link score", "The average 0 to 100 link score of pages in this section."),
    "avg_score_pre": ("Average link score before", "Average link score on the old site."),
    "avg_score_post": ("Average link score after", "Average link score on the new site."),
    "avg_inlinks": ("Average links in", "The average number of links pointing at pages in this section."),
    "avg_inlinks_pre": ("Average links in before", "On the old site."),
    "avg_inlinks_post": ("Average links in after", "On the new site."),
    "median_depth": ("Typical clicks from home", "The middle value of clicks from home for pages in this section."),
    "median_depth_pre": ("Typical clicks from home before", "On the old site."),
    "median_depth_post": ("Typical clicks from home after", "On the new site."),
    "share_change_pts": ("Change in share (points)", "Share after minus share before. Negative means the section lost link strength to the rest of the site."),
    # ---- linking plan
    "pr_x_avg_before": ("Strength before (× avg)", "Link strength before your changes, as a multiple of the average page."),
    "pr_x_avg_after": ("Strength after (× avg)", "Link strength after your changes."),
    "change_pct": ("Change (%)", "Percent change in link strength caused by your changes."),
    "percentile_before": ("Rank before", "Rank among the site's pages before your changes."),
    "percentile_after": ("Rank after", "Rank among the site's pages after your changes."),
    "inlinks_before": ("Links in before", "Links pointing at the page before your changes."),
    "inlinks_after": ("Links in after", "Links pointing at the page after your changes."),
    "source_pr_x_avg": ("Linking page strength (× avg)", "How strong the page that would hold the new link is."),
    "source_outlinks": ("Linking page's links out", "How many links the linking page already has. The more it has, the less each one passes."),
    "link_value": ("Value per link (est.)", "A rough score for how much strength a new link from this page would pass: its strength divided by its links out. Higher is better."),
    "links": ("Links", "How many links point at this page."),
    "similarity": ("Similarity (0 to 1)", "How close in meaning the linking page is to the target page. Higher is more related. The scale depends on the method, so compare suggestions with each other rather than against a fixed number."),
    "relevance": ("How related", "Very close, Close or Related: where this page ranks among all pages for similarity to the target."),
    "suggested_anchor": ("Suggested link text", "The words to link. Taken from wording already on the linking page whenever possible, so the link reads naturally."),
    "anchor_basis": ("Where the link text came from", "Whether the phrase was already on the linking page, or had to be borrowed from the target's heading and placed by hand."),
    "context": ("Sentence it appears in", "The sentence on the linking page that contains the suggested link text. Add the link there."),
    "anchor_note": ("Link text warning", "Appears when the suggested wording is already the most common link text pointing at the target, so repeating it adds little. Vary the wording."),
    "priority": ("Priority score", "Similarity adjusted for how much strength the linking page can pass. Higher means a better suggestion."),
    "method": ("How matched", "Embeddings (page meaning from a language model) or Text (TF-IDF: shared words and phrases)."),
}

FLAG_LABELS = {"LOSER": "Lost strength", "GAINER": "Gained strength", "DEEPER": "Pushed deeper",
               "UNREACHABLE": "Cut off from home", "NO_INLINKS": "No links in"}
FLAG_HELP = {
    "Lost strength": "The page kept less than your chosen share of its old link strength, or fell sharply in rank.",
    "Gained strength": "The page gained a lot of link strength compared with before.",
    "Pushed deeper": "It now takes noticeably more clicks from the home page to reach it.",
    "Cut off from home": "The page used to be reachable from the home page and no longer is.",
    "No links in": "No page on the new site links to it.",
}

SUMMARY: Dict[str, Tuple[str, str]] = {
    "pages_pre": ("Pages on the old site", "Pages found in the old crawl."),
    "pages_post": ("Pages on the new site", "Pages found in the new crawl."),
    "old_urls_compared": ("Old pages compared", "Old pages that could be indexed and were checked against the new site."),
    "matched_to_post_crawl": ("Old pages found on the new site", "Old pages whose new address was found in the new crawl."),
    "old_urls_with_no_destination": ("Old pages with no new home", "Old pages with no matching page in the new crawl, so their link strength has nowhere to go."),
    "share_of_pre_equity_with_no_destination": ("Share of old strength with no new home", "The part of the old site's total link strength that sat on those pages."),
    "losers": ("Pages that lost strength", "Pages flagged as having lost a lot of link strength."),
    "gainers": ("Pages that gained strength", "Pages flagged as having gained a lot of link strength."),
    "deeper_pages": ("Pages pushed deeper or cut off", "Pages that are now more clicks from home, or can no longer be reached from the home page."),
    "median_ratio": ("Typical share of strength kept", "The middle page's 'Kept' value. 1.0 means a typical page held steady."),
    "rank_stability_spearman": ("How similar the page ranking is (0 to 1)", "How closely the order of pages by link strength matches before and after. 1 means identical order; low numbers mean the pecking order was reshuffled."),
    "post_orphans_or_unreachable": ("Pages with no way in", "Pages on the new site that nothing links to, or that can't be reached from the home page."),
}


def label_for(col: str) -> str:
    if col in COLUMNS:
        return COLUMNS[col][0]
    return col.replace("_", " ").strip().capitalize()


def help_for(col: str) -> str:
    return COLUMNS[col][1] if col in COLUMNS else ""


def _friendly_flags(v) -> str:
    if not isinstance(v, str) or not v:
        return ""
    return ", ".join(FLAG_LABELS.get(t.strip(), t.strip()) for t in v.split(","))


def friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with plain-English column names (and flag words)."""
    d = df.copy()
    if "flags" in d.columns:
        d["flags"] = d["flags"].map(_friendly_flags)
    d.columns = [label_for(c) for c in d.columns]
    return d


def friendly_summary(summary: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for k, v in summary.items():
        if k.startswith("top_") and "still_in_top" in k:
            n = k.split("_")[1]
            rows.append((f"Top {n} pages still in the top {n}",
                         f"Of the {n} best-linked old pages, the share that are still in the top {n} on the new site.", v))
        else:
            lab, hlp = SUMMARY.get(k, (k.replace("_", " ").capitalize(), ""))
            rows.append((lab, hlp, v))
    return pd.DataFrame(rows, columns=["Measure", "What it means", "Value"])


def column_config(df: pd.DataFrame, st) -> dict:
    """Streamlit column_config with a hover tooltip on every header of an already-friendly frame."""
    inverse = {label_for(c): c for c in COLUMNS}
    cfg = {}
    for lab in df.columns:
        src = inverse.get(lab)
        hlp = help_for(src) if src else ""
        if lab == "What changed":
            hlp += " " + " ".join(f"{k}: {v}" for k, v in FLAG_HELP.items())
        if hlp:
            cfg[lab] = st.column_config.Column(label=lab, help=hlp)
    return cfg


def glossary_table() -> pd.DataFrame:
    """Everything in one table for the Glossary tab (one row per label)."""
    seen, rows = set(), []
    for col, (lab, hlp) in COLUMNS.items():
        if lab in seen:
            continue
        seen.add(lab)
        rows.append((lab, hlp, col))
    for lab, hlp in FLAG_HELP.items():
        rows.append((f"Flag: {lab}", hlp, ""))
    return pd.DataFrame(rows, columns=["Name", "What it means", "Technical name (CLI and code)"])
