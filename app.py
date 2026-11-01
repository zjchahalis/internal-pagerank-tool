"""Streamlit app: score internal PageRank from Screaming Frog crawls and measure a migration.

Run locally:   streamlit run app.py
Deploy free:   push this folder to GitHub, then create an app on Streamlit Community Cloud.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

import glossary as gl
import ipr_core as ipr
import ipr_related as rel

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_data")

st.set_page_config(page_title="Internal PageRank Comparison Tool", layout="wide")


# ----------------------------------------------------------------------------- helpers
def show(df: pd.DataFrame, **kw) -> None:
    """Show a table with plain-English headers, each with a hover tooltip."""
    d = df.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].round(3)
    f = gl.friendly(d)
    cfg = gl.column_config(f, st)
    try:
        st.dataframe(f, width="stretch", hide_index=True, column_config=cfg, **kw)
    except TypeError:
        st.dataframe(f, use_container_width=True, hide_index=True, column_config=cfg, **kw)


def friendly_zip(frames: dict) -> bytes:
    """Zip of reports with plain-English column names."""
    out = {}
    for k, v in frames.items():
        out[k] = v if k == "summary" else gl.friendly(v)
    return ipr.reports_to_zip(out)


@st.cache_data(show_spinner=False, max_entries=8)
def read_bytes(data: bytes, all_cols: bool = False) -> pd.DataFrame:
    return ipr.read_sf_csv(io.BytesIO(data), usecols_all=all_cols)


@st.cache_data(show_spinner="Scoring link graph...", max_entries=8)
def build_scores(inl: bytes, internal: bytes | None, opts: tuple):
    o = dict(opts)
    g = ipr.build_graph(
        read_bytes(inl), read_bytes(internal) if internal else None,
        follow_only=o["follow_only"], exclude_positions=list(o["exclude_positions"]) or None,
        position_weights=dict(o["weights"]) or None, resolve_redirects=o["resolve"], home=o["home"] or None)
    return g, ipr.score_graph(g, o["damping"])


@st.cache_data(show_spinner="Reading embeddings...", max_entries=4)
def load_vectors(data: bytes):
    return rel.load_embeddings(rel.read_embeddings_csv(io.BytesIO(data)))


@st.cache_data(show_spinner="Reading page text...", max_entries=4)
def load_text_index(internal: bytes | None, text: bytes | None):
    pages = rel.build_page_text(read_bytes(internal, True) if internal else None,
                                read_bytes(text, True) if text else None)
    return rel.build_text_index(pages) if len(pages) else None


@st.cache_data(show_spinner=False, max_entries=4)
def load_anchors(inl: bytes):
    return rel.anchor_counts(rel.read_anchor_table(io.BytesIO(inl)))


def get_file(label: str, key: str, sample_name: str | None, help: str | None = None) -> bytes | None:
    up = st.file_uploader(label, type=["csv"], key=key, help=help)
    if up is not None:
        return up.getvalue()
    if st.session_state.get("use_sample") and sample_name:
        path = os.path.join(SAMPLE, sample_name)
        if os.path.exists(path):
            st.caption(f"Using sample file: {sample_name}")
            return open(path, "rb").read()
    return None


def fmt(v):
    return f"{v:,.3f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Graph settings")
st.sidebar.checkbox("Use the bundled sample data", key="use_sample",
                    help="A made-up shop before and after a replatform, so you can try every tab.")
follow_only = st.sidebar.checkbox("Ignore nofollow links", value=True)
positions = st.sidebar.multiselect("Ignore links in these positions", ["Navigation", "Header", "Footer", "Aside", "Content"],
                                   help="Screaming Frog classifies each link by where it sits on the page.")
preset = st.sidebar.selectbox("Link weighting", ["Every link counts the same", "Weight by position (reasonable surfer style)", "Custom"])
weights_text = ""
if preset.startswith("Weight"):
    weights_text = "content=1,navigation=0.5,header=0.5,aside=0.3,footer=0.2,default=1"
elif preset == "Custom":
    weights_text = st.sidebar.text_input("position=weight, comma separated", "content=1,navigation=0.5,footer=0.2,default=1")
damping = st.sidebar.slider("Damping factor", 0.5, 0.95, 0.85, 0.01)
resolve = st.sidebar.checkbox("Pass equity through redirects", value=True,
                              help="Links that point at a redirecting URL are credited to the final destination.")
home = st.sidebar.text_input("Home page URL (optional)", "", help="Detected automatically if blank. Used for click depth.")
st.sidebar.info("Client crawls are confidential. A hosted copy of this app sends uploads to its host. "
                "Run it locally for client data, or use the hosted copy with the sample data.")

OPTS = (("follow_only", follow_only), ("exclude_positions", tuple(positions)),
        ("weights", tuple(sorted(ipr.parse_position_weights(weights_text).items()))),
        ("resolve", resolve), ("damping", damping), ("home", home))

# ----------------------------------------------------------------------------- page
st.title("Internal PageRank Comparison Tool")
st.write("Score how link equity flows through your own site, compare it before and after a migration, "
         "and test an internal linking plan before you ship it. Works from Screaming Frog exports.")

t_how, t_score, t_cmp, t_plan, t_gloss = st.tabs(["1. Export from Screaming Frog", "2. Score a crawl",
                                                  "3. Compare before and after", "4. Test a linking plan", "Glossary"])

# ---- tab 1
with t_how:
    st.markdown("""
**Export two files from each crawl.** Use CSV.

1. **All Inlinks**: *Bulk Export > Links > All Inlinks*. This is the link graph (Source, Destination, Follow, Link Position).
2. **Internal: All**: the *Internal* tab, filter *All*, then *Export*. This gives status codes, indexability, redirect targets and crawl depth.

**Crawl settings that matter**

- Turn on JavaScript rendering if the site builds navigation or links in the browser.
- Crawl the same scope before and after (same subdomains, same limits).
- To catch orphan pages, enable crawling of linked XML sitemaps so pages with no links still appear in the Internal export.

**For a migration**, also bring your **redirect map**: a CSV with `Source` and `Target` columns (old URL, new URL). Without one, old and new URLs are matched when they are identical, or by path if you tick *match by path* (staging versus production).

**Optional: page embeddings for related-link suggestions (tab 4).** The best suggestions come from page embeddings made with a local model such as EmbeddingGemma, so page meaning is compared instead of just words. Create the file on your own machine:

1. Install [Ollama](https://ollama.com) and run `ollama pull embeddinggemma`.
2. Export **Internal: All** from Screaming Frog (it includes Title 1, H1-1 and Meta Description 1). For link text drawn from body copy, also export a CSV of Address plus the main content text (a Screaming Frog Custom Extraction works).
3. Run `python embed_pages.py --internal internal_all.csv --page-text page_text.csv --out embeddings.csv`.
4. Upload `embeddings.csv` in tab 4. It needs an **Address** column plus one number column per dimension (or one column holding the whole vector). If your Screaming Frog version can export embeddings, that file works too.

No embeddings? Tab 4 falls back to comparing shared words and phrases (TF-IDF) from titles, H1s, meta descriptions and any page text you upload. It runs with no setup and is a good first pass.

**Reading the output.** Hover over any column heading or metric for a plain-English explanation, or open the Glossary tab. The key number is **Link strength (× average)**: 1.0 is an average page, 3.0 is three times average, 0.2 is a fifth of average. Values above 1.0 are normal. **PageRank (0 to 1)** is the true PageRank, which gets tiny on big sites, so compare the strength column instead.
""")

# ---- tab 2
with t_score:
    c1, c2 = st.columns(2)
    with c1:
        inl = get_file("All Inlinks CSV", "s_inl", "post_all_inlinks.csv")
    with c2:
        itn = get_file("Internal: All CSV (recommended)", "s_int", "post_internal_all.csv")
    if inl:
        try:
            g, sc = build_scores(inl, itn, OPTS)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build the graph: {e}")
            st.stop()
        m = st.columns(5)
        m[0].metric("Pages in graph", fmt(g.stats["nodes"]), help="How many pages were found in the crawl and scored.")
        m[1].metric("Internal links used", fmt(g.stats["edges_used"]), help="Links between pages that were counted, after your sidebar settings (for example nofollow links ignored).")
        m[2].metric("Pages with no links in", fmt(int(sc.orphan.sum())), help="Pages that no other page links to.")
        m[3].metric("Cut off from home", fmt(int(sc.unreachable.sum())), help="Pages you can not reach from the home page by clicking links.")
        m[4].metric("Links to broken pages", fmt(g.stats["broken_link_rows"]), help="Links that point at pages returning an error (404, 500 and so on). They pass no strength.")
        st.caption(f"Home page: {g.home}")
        show_idx = st.checkbox("Indexable pages only", value=True, key="s_idx")
        view = sc[sc.indexable] if show_idx else sc
        st.subheader("Best-linked pages")
        st.caption("Link strength compares each page with an average page on your site: 1.0 is average, so values above 1.0 are normal. Hover over a heading for details.")
        show(view[["url", "pr_x_avg", "pr", "score", "percentile", "inlinks", "outlinks", "depth"]].head(100))
        st.subheader("Link strength by section")
        seg_txt = st.text_area("Optional: define sections as Name=regex (one per line, matched against the URL path)",
                               "", key="s_seg", height=80)
        seg = ipr.segment_rollup(view, ipr.parse_segments(seg_txt) or None)
        show(seg)
        st.subheader("Click depth")
        st.caption("How many pages sit at each number of clicks from the home page. Blank means it can not be reached.")
        st.bar_chart(view.depth.value_counts(dropna=False).sort_index().rename_axis("Clicks from home").rename("Pages"))
        with st.expander("Pages with no links in, or cut off from home"):
            show(sc[(sc.orphan | sc.unreachable) & sc.indexable][["url", "pr_x_avg", "inlinks", "depth"]])
        with st.expander("Internal links pointing at broken pages"):
            show(g.broken.groupby(["destination", "status"]).size().rename("links").reset_index()
                 .sort_values("links", ascending=False) if len(g.broken) else g.broken)
        st.download_button("Download scores and reports (zip)", friendly_zip(
            {"scores": sc, "sections": seg, "orphans_unreachable": sc[sc.orphan | sc.unreachable],
             "broken_links": g.broken}), "internal_pagerank_scores.zip", mime="application/zip")
    else:
        st.info("Upload your All Inlinks export to start, or tick the sample data box in the sidebar.")

# ---- tab 3
with t_cmp:
    st.write("Upload the crawl from before the migration and the crawl from after (or from staging).")
    a, b = st.columns(2)
    with a:
        st.markdown("**Before (old site)**")
        pre_i = get_file("All Inlinks CSV", "c_pre_inl", "pre_all_inlinks.csv")
        pre_n = get_file("Internal: All CSV", "c_pre_int", "pre_internal_all.csv")
    with b:
        st.markdown("**After (new site or staging)**")
        post_i = get_file("All Inlinks CSV", "c_post_inl", "post_all_inlinks.csv")
        post_n = get_file("Internal: All CSV", "c_post_int", "post_internal_all.csv")
    map_b = get_file("Redirect map CSV (Source, Target). Optional.", "c_map", "redirect_map.csv")
    o1, o2, o3, o4 = st.columns(4)
    by_path = o1.checkbox("Match by path (ignore host)", help="Use when staging has a different hostname but the same paths.")
    strip = o2.checkbox("Ignore trailing slash differences")
    loser_ratio = o3.slider("Flag a page that keeps less than", 0.1, 0.9, 0.5, 0.05,
                            help="Share of its old link strength. 0.5 means it lost half or more.")
    depth_inc = o4.slider("Flag when it gets this many clicks deeper", 1, 5, 2, help="Clicks from the home page.")
    seg_txt = st.text_area("Optional: define sections as Name=regex so renamed directories line up "
                           "(for example Products=^/products/|^/shop/[^/]+/[^/]+$)", "", key="c_seg", height=80)

    if pre_i and post_i:
        try:
            pre_g, pre = build_scores(pre_i, pre_n, OPTS)
            post_g, post = build_scores(post_i, post_n, OPTS)
            mapping = ipr.load_mapping(read_bytes(map_b, True)) if map_b else None
            res = ipr.compare(pre, post, mapping, ignore_host=by_path,
                              trailing_slash="strip" if strip else "keep", loser_ratio=loser_ratio,
                              depth_increase=depth_inc, segments=ipr.parse_segments(seg_txt) or None)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not compare the crawls: {e}")
            st.stop()
        sm = res["summary"]
        m = st.columns(5)
        m[0].metric("Old pages compared", fmt(sm["old_urls_compared"]), help=gl.SUMMARY["old_urls_compared"][1])
        m[1].metric("Pages that lost strength", fmt(sm["losers"]), help=gl.SUMMARY["losers"][1])
        m[2].metric("Old pages with no new home", fmt(sm["old_urls_with_no_destination"]),
                    f"{sm['share_of_pre_equity_with_no_destination']:.1%} of old link strength", delta_color="off",
                    help=gl.SUMMARY["old_urls_with_no_destination"][1] + " The line underneath is the share of the old site's total link strength that sat on them.")
        m[3].metric("Pages pushed deeper or cut off", fmt(sm["deeper_pages"]), help=gl.SUMMARY["deeper_pages"][1])
        m[4].metric("Pages with no way in", fmt(sm["post_orphans_or_unreachable"]), help=gl.SUMMARY["post_orphans_or_unreachable"][1])
        tk = [k for k in sm if k.startswith("top_")][0]
        st.caption(f"Of the top pages before, {sm[tk]:.0%} are still in the top after. "
                   f"Page ranking similarity (0 to 1, higher is more alike): {sm['rank_stability_spearman']:.2f}. "
                   f"Typical share of strength kept: {sm['median_ratio']:.2f}.")

        cols = ["new_url", "old_urls", "old_count", "pre_pr_x_avg", "post_pr_x_avg", "ratio", "pre_pr", "post_pr",
                "pre_inlinks", "post_inlinks", "pre_depth", "post_depth", "flags"]
        st.caption("Hover over any heading for what it means. Strength is a multiple of the average page (1.0 = average). "
                   "Merged pages show the combined strength of their old pages.")
        st.subheader("Pages that lost the most strength")
        st.caption("Sorted by how much strength was lost, so important pages surface first. "
                   "When several old pages merge into one, their strength is added together before comparing.")
        show(res["losers"][cols].head(200))
        st.subheader("Pages that gained the most strength")
        show(res["gainers"][cols].head(100))
        st.subheader("Before versus after")
        st.caption("Each dot is a page. Dots on the diagonal kept their strength; dots below it lost strength.")
        sc_df = res["pairs"][["pre_pr_x_avg", "post_pr_x_avg"]].rename(columns={
            "pre_pr_x_avg": "Strength before (× avg)", "post_pr_x_avg": "Strength after (× avg)"})
        st.scatter_chart(sc_df, x="Strength before (× avg)", y="Strength after (× avg)")
        st.subheader("Old pages with nowhere to go")
        st.caption("These old pages carried link strength but have no matching page in the new crawl.")
        show(res["unmapped_old"])
        st.subheader("Pages pushed deeper or cut off from the home page")
        show(res["deeper"][cols])
        st.subheader("Pages on the new site with no way in")
        show(res["orphans_post"])
        st.subheader("New pages with no old equivalent")
        show(res["new_pages"].head(100))
        st.subheader("Link strength by section")
        show(res["segments"])
        with st.expander("Internal links to broken pages on the new site"):
            show(post_g.broken.groupby(["destination", "status"]).size().rename("links").reset_index()
                 .sort_values("links", ascending=False) if len(post_g.broken) else post_g.broken)
        reports = {k: v for k, v in res.items() if k != "summary"}
        reports["summary"] = gl.friendly_summary(sm)
        st.download_button("Download all reports (zip)", friendly_zip(reports),
                           "migration_internal_pagerank.zip", mime="application/zip")
    else:
        st.info("Upload both All Inlinks files to compare, or tick the sample data box in the sidebar.")

# ---- tab 4
with t_plan:
    st.write("Test an internal linking plan on a crawl before you ship it. Pick the pages you want to lift and the tool "
             "suggests which related pages should link to them, with link text, then shows who gains and who loses "
             "link strength. Or bring your own list of link changes.")
    c1, c2 = st.columns(2)
    with c1:
        p_inl = get_file("All Inlinks CSV", "p_inl", "post_all_inlinks.csv")
    with c2:
        p_int = get_file("Internal: All CSV", "p_int", "post_internal_all.csv")
    mode = st.radio("Plan source", ["Suggest related links (embeddings or text)", "Suggest by link strength only",
                                    "Upload my own link changes"], horizontal=True,
                    help="Related links pick pages that are about the same thing as the target, then order them by "
                         "how much strength they can pass. Strength only ignores meaning and is a weaker plan.")
    if p_inl:
        try:
            g, base = build_scores(p_inl, p_int, OPTS)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build the graph: {e}")
            st.stop()
        changes, targets = None, []
        sample_on = bool(st.session_state.get("use_sample"))
        default_t = "https://www.example-shop.com/shop/accessories/" if sample_on else ""
        if mode.startswith("Suggest related"):
            st.markdown("**How should pages be matched?**")
            e1, e2 = st.columns(2)
            with e1:
                emb_b = get_file("Page embeddings CSV (best): Address plus one column per dimension", "p_emb",
                                 "post_embeddings.csv",
                                 help="Made with a local model such as EmbeddingGemma. See tab 1 for how to create it.")
            with e2:
                txt_b = get_file("Page text CSV (optional): Address plus body text", "p_txt", "post_page_text.csv",
                                 help="Lets the tool suggest link text from words already on the linking page. "
                                      "Without it, only titles, H1s and meta descriptions from Internal: All are used.")
            index, tix, err = None, None, None
            try:
                tix = load_text_index(p_int, txt_b)
                if emb_b:
                    index = load_vectors(emb_b)
                    cov = len([u for u in base.url if index.has(u)]) / max(len(base), 1)
                    st.caption(f"Using embeddings: {len(index.urls):,} pages, {index.dims} dimensions"
                               + (f", {index.dropped} rows skipped" if index.dropped else "")
                               + f". {cov:.0%} of crawled pages have a vector.")
                    if cov < 0.9:
                        st.warning("Many crawled pages have no embedding, so they cannot be suggested. "
                                   "Check the Address values match the crawl.")
                elif tix is not None:
                    index = tix
                    st.caption("No embeddings uploaded, so pages are matched on shared words and phrases (TF-IDF). "
                               "Good for a first pass; embeddings understand meaning and usually do better.")
                else:
                    err = "Upload an embeddings CSV, or a crawl with page titles (Internal: All) or a page text CSV, to match pages."
            except Exception as e:  # noqa: BLE001
                err = f"Could not read the matching data: {e}"
            if err:
                st.info(err)
            tt = st.text_area("Target URLs to lift, one per line", default_t, height=100)
            c1_, c2_, c3_ = st.columns(3)
            k = c1_.slider("Suggested links per target", 1, 20, 5)
            pct = c2_.slider("Only the most related pages", 1, 50, 10, format="top %d%%",
                             help="Of the pages that could still link to the target, how many are considered. "
                                  "Smaller means stricter.")
            eq = c3_.slider("Weight on link strength", 0.0, 1.0, 0.5, 0.1,
                            help="0 orders purely by how related the page is. 1 leans on how much strength the page can pass. "
                                 "Relevance always decides who is eligible.")
            with st.expander("More options"):
                same = st.checkbox("Only suggest sources in the same section")
                min_sim = st.slider("Also require at least this similarity", 0.0, 1.0, 0.0, 0.05,
                                    help="An extra floor on top of the 'most related' setting. Scales differ by model.")
                max_out = st.number_input("Skip linking pages that already have more links than (0 = no limit)", 0, 5000, 0)
            targets = [t.strip() for t in tt.splitlines() if t.strip()]
            if targets and index is not None:
                changes = rel.suggest_related_links(
                    g, base, targets, index, text_index=tix, anchors=load_anchors(p_inl) if p_inl else None, k=k,
                    min_percentile=100 - pct, min_similarity=min_sim, equity_weight=eq, same_segment=same,
                    max_source_outlinks=int(max_out) or None)
                if changes.attrs.get("skipped"):
                    st.warning("No embedding or text found for: " + ", ".join(changes.attrs["skipped"]))
                if len(changes):
                    st.caption("Review before applying: link text comes from wording already on the linking page where possible. "
                               "Hover over a heading for details.")
                    show(changes)
                else:
                    st.info("No suggestions. Try a larger share of pages, or lower the similarity floor.")
        elif mode.startswith("Suggest by link"):
            tt = st.text_area("Target URLs to lift, one per line", default_t, height=100)
            k = st.slider("Suggested links per target", 1, 20, 5)
            same = st.checkbox("Only suggest sources in the same section")
            targets = [t.strip() for t in tt.splitlines() if t.strip()]
            if targets:
                changes = ipr.suggest_links(g, base, targets, k=k, same_segment=same)
                st.caption("Ranked by how strong the linking page is divided by how many links it already has. "
                           "This ignores whether the pages are related, so treat it as a rough guide.")
                show(changes)
        else:
            ch_b = get_file("Changes CSV: action (add/remove), source, destination, optional weight", "p_ch",
                            "scenario_changes.csv")
            if ch_b:
                changes = read_bytes(ch_b, True)
                ac, dc = ipr._col(changes, "action"), ipr._col(changes, "destination")
                targets = changes.loc[changes[ac].astype(str).str.lower() == "add", dc].tolist()
                show(changes)
        if changes is not None and len(changes):
            scen = ipr.score_graph(ipr.apply_scenario(g, changes), damping)
            rep = ipr.scenario_report(base, scen, targets)
            if "summary" in rep:
                s = rep["summary"]
                m = st.columns(3)
                m[0].metric("Target pages", fmt(s["targets"]))
                m[1].metric("Link strength gained by targets", f"{s['target_pr_x_avg_gain']:+.2f}", help="Total change in Link strength (× average) across the pages you want to lift.")
                m[2].metric("Strength taken from other pages", f"{s['others_pr_x_avg_change']:+.2f}", help="Total change in Link strength (× average) across all other pages. Link strength is shared, so a gain for one page is a loss somewhere else.")
                st.subheader("Target pages, before and after")
                show(rep["targets"])
            st.subheader("Pages that lose the most")
            show(rep["losers"])
            st.subheader("Pages that gain the most")
            show(rep["gainers"])
            st.download_button("Download plan and results (zip)", friendly_zip(
                {"scenario_changes": changes, "scenario_all": rep["all"], "scenario_gainers": rep["gainers"],
                 "scenario_losers": rep["losers"]}), "linking_plan_results.zip", mime="application/zip")
    else:
        st.info("Upload a crawl to start, or tick the sample data box in the sidebar.")

# ---- glossary
with t_gloss:
    st.write("Every column and measure in this tool, in plain English. You can also hover over any table heading or metric.")
    st.markdown("**The one idea behind most of it:** pages that lots of strong pages link to are 'well linked'. "
                "The tool measures how well linked each page is, then shows what changed between two versions of your site.")
    gdf = gl.glossary_table()
    q = st.text_input("Search", "", key="gloss_q")
    if q:
        mask = gdf.apply(lambda r: q.lower() in " ".join(map(str, r)).lower(), axis=1)
        gdf = gdf[mask]
    try:
        st.dataframe(gdf, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(gdf, use_container_width=True, hide_index=True)
