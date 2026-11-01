import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ipr_core as ipr  # noqa: E402

SAMPLE = os.path.join(ROOT, "sample_data")


def sample(name, all_cols=False):
    return ipr.read_sf_csv(os.path.join(SAMPLE, name), usecols_all=all_cols)


def tiny_graph(edges, nodes=None):
    nodes = nodes or sorted({u for e in edges for u in e[:2]})
    idx = {u: i for i, u in enumerate(nodes)}
    meta = pd.DataFrame({"indexable": True, "status": 200.0, "sf_depth": np.nan, "sf_link_score": np.nan}, index=nodes)
    return ipr.LinkGraph(nodes, np.array([idx[e[0]] for e in edges]), np.array([idx[e[1]] for e in edges]),
                         np.array([e[2] if len(e) > 2 else 1.0 for e in edges]), meta, nodes[0],
                         pd.DataFrame(columns=["source", "destination", "status"]))


def test_pagerank_matches_networkx():
    rng = np.random.default_rng(3)
    n = 60
    edges = {(f"/p{a}", f"/p{b}") for a, b in rng.integers(0, n, (300, 2)) if a != b}
    g = tiny_graph([(a, b) for a, b in edges], nodes=[f"/p{i}" for i in range(n)])
    mine = ipr.pagerank(g, 0.85)
    G = nx.DiGraph()
    G.add_nodes_from(g.nodes)
    G.add_edges_from(edges)
    ref = nx.pagerank(G, alpha=0.85, tol=1e-12)
    assert abs(mine.sum() - 1) < 1e-9
    assert np.allclose(mine, [ref[u] for u in g.nodes], atol=1e-8)


def test_weighted_pagerank_matches_networkx():
    edges = [("/a", "/b", 3.0), ("/a", "/c", 1.0), ("/b", "/a", 1.0), ("/c", "/a", 1.0), ("/c", "/b", 2.0)]
    g = tiny_graph(edges)
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    ref = nx.pagerank(G, alpha=0.85, weight="weight", tol=1e-12)
    assert np.allclose(ipr.pagerank(g), [ref[u] for u in g.nodes], atol=1e-8)


def make_inlinks(rows):
    return pd.DataFrame(rows, columns=["Type", "Source", "Destination", "Status Code", "Follow", "Link Position"])


def make_internal(rows):
    return pd.DataFrame(rows, columns=["Address", "Content Type", "Status Code", "Indexability", "Redirect URL", "Crawl Depth"])


def test_build_graph_filters_redirects_and_broken():
    base = "https://x.com"
    inl = make_inlinks([
        ["Hyperlink", f"{base}/", f"{base}/a", "200", "true", "Content"],
        ["Hyperlink", f"{base}/", f"{base}/old", "301", "true", "Navigation"],      # redirects to /b
        ["Hyperlink", f"{base}/", f"{base}/gone", "404", "true", "Content"],
        ["Hyperlink", f"{base}/", f"{base}/nf", "200", "false", "Content"],         # nofollow
        ["Hyperlink", f"{base}/a", f"{base}/#frag", "200", "true", "Footer"],       # fragment -> home
        ["Image", f"{base}/a", f"{base}/img.png", "200", "true", "Content"],        # not a hyperlink
    ])
    itn = make_internal([
        [f"{base}/", "text/html", "200", "Indexable", "", "0"], [f"{base}/a", "text/html", "200", "Indexable", "", "1"],
        [f"{base}/b", "text/html", "200", "Indexable", "", "1"], [f"{base}/nf", "text/html", "200", "Indexable", "", "1"],
        [f"{base}/old", "text/html", "301", "Non-Indexable", f"{base}/b", ""],
        [f"{base}/gone", "text/html", "404", "Non-Indexable", "", ""], [f"{base}/img.png", "image/png", "200", "", "", ""],
    ])
    g = ipr.build_graph(inl, itn)
    edges = set(zip((g.nodes[i] for i in g.src), (g.nodes[i] for i in g.dst)))
    assert (f"{base}/", f"{base}/b") in edges           # redirect resolved
    assert (f"{base}/", f"{base}/a") in edges
    assert (f"{base}/", f"{base}/nf") not in edges      # nofollow ignored
    assert (f"{base}/a", f"{base}/") in edges           # fragment stripped
    assert not any("img.png" in e[1] for e in edges)
    assert f"{base}/old" not in g.nodes and f"{base}/gone" not in g.nodes
    assert list(g.broken.destination) == [f"{base}/gone"]
    g2 = ipr.build_graph(inl, itn, follow_only=False)
    assert (f"{base}/", f"{base}/nf") in set(zip((g2.nodes[i] for i in g2.src), (g2.nodes[i] for i in g2.dst)))
    g3 = ipr.build_graph(inl, itn, exclude_positions=["footer"])
    assert (f"{base}/a", f"{base}/") not in set(zip((g3.nodes[i] for i in g3.src), (g3.nodes[i] for i in g3.dst)))
    assert g.home == f"{base}/"


def test_depth_orphan_and_unreachable():
    g = tiny_graph([("/", "/a"), ("/a", "/b"), ("/x", "/y"), ("/y", "/x")], nodes=["/", "/a", "/b", "/x", "/y", "/o"])
    s = ipr.score_graph(g).set_index("url")
    assert s.loc["/b", "depth"] == 2 and s.loc["/a", "depth"] == 1
    assert s.loc["/x", "unreachable"] and not s.loc["/x", "orphan"]      # linked, but cut off from home
    assert s.loc["/o", "orphan"] and s.loc["/o", "unreachable"]
    assert not s.loc["/", "orphan"]


def test_scenario_raises_target_and_conserves_mass():
    g = tiny_graph([("/", "/a"), ("/", "/b"), ("/a", "/"), ("/b", "/"), ("/a", "/c")], nodes=["/", "/a", "/b", "/c", "/d"])
    base = ipr.score_graph(g)
    ch = pd.DataFrame({"action": ["add"], "source": ["/"], "destination": ["/d"], "weight": [1]})
    scen = ipr.score_graph(ipr.apply_scenario(g, ch))
    rep = ipr.scenario_report(base, scen, ["/d"])
    assert rep["summary"]["target_pr_x_avg_gain"] > 0
    assert abs(scen.pr.sum() - 1) < 1e-9
    rm = pd.DataFrame({"action": ["remove"], "source": ["/a"], "destination": ["/c"]})
    after = ipr.score_graph(ipr.apply_scenario(g, rm)).set_index("url")
    assert after.loc["/c", "pr"] < base.set_index("url").loc["/c", "pr"]


def test_suggest_links_skips_existing_and_ranks_by_value():
    g = tiny_graph([("/", "/a"), ("/", "/b"), ("/a", "/"), ("/b", "/"), ("/a", "/t")], nodes=["/", "/a", "/b", "/t"])
    sc = ipr.score_graph(g)
    sug = ipr.suggest_links(g, sc, ["/t"], k=5)
    assert "/a" not in set(sug.source)            # already links to /t
    assert set(sug.source) == {"/", "/b"}


def test_sample_migration_story():
    pre = ipr.score_graph(ipr.build_graph(sample("pre_all_inlinks.csv"), sample("pre_internal_all.csv")))
    post = ipr.score_graph(ipr.build_graph(sample("post_all_inlinks.csv"), sample("post_internal_all.csv")))
    mp = ipr.load_mapping(sample("redirect_map.csv", True))
    res = ipr.compare(pre, post, mp)
    s = res["summary"]
    assert s["old_urls_with_no_destination"] == 5                     # five removed posts left out of the map
    pairs = res["pairs"].set_index("new_url")
    guide = pairs.loc["https://www.example-shop.com/resources/migration-guide"]
    assert guide.old_count == 3                                       # three posts merged into one guide
    merged_pre = pre.set_index("url").loc[[f"https://www.example-shop.com/blog/post-{i}" for i in (1, 2, 3)], "pr_x_avg"].sum()
    assert abs(guide.pre_pr_x_avg - merged_pre) < 1e-9                  # equity summed, not averaged
    acc = pairs.loc["https://www.example-shop.com/shop/accessories/"]
    assert "UNREACHABLE" in acc["flags"]                                 # orphaned section is caught
    assert (res["orphans_post"].new_url.str.contains("accessories")).all() and len(res["orphans_post"]) == 41
    res_hub = pairs.loc["https://www.example-shop.com/resources/"]
    assert res_hub.post_depth - res_hub.pre_depth >= 2 and "DEEPER" in res_hub["flags"]   # buried blog hub
    assert (res["losers"].new_url.str.contains("/shop/shoes/shoes-")).any()           # products lost related links
    seg = ipr.segment_compare(ipr.segment_rollup(pre[pre.indexable], ipr.parse_segments(open(os.path.join(SAMPLE, "segments.txt")).read())),
                              ipr.segment_rollup(post[post.indexable], ipr.parse_segments(open(os.path.join(SAMPLE, "segments.txt")).read())))
    prod = seg.set_index("segment").loc["Products"]
    assert prod.pages_pre == 120 and prod.pages_post == 120 and prod.share_change_pts < 0   # products lost share


def test_ignore_host_matches_staging_by_path():
    pre = pd.DataFrame({"url": ["https://old.com/a", "https://old.com/b"], "pr_x_avg": [2.0, 1.0], "percentile": [90.0, 50.0],
                        "inlinks": [3, 2], "depth": [1.0, 2.0], "indexable": True, "orphan": False, "unreachable": False,
                        "pr": [.2, .1], "score": [90., 50.]})
    post = pre.assign(url=["https://staging.new.com/a", "https://staging.new.com/b"], pr_x_avg=[1.0, 1.0])
    r = ipr.compare(pre, post, None, ignore_host=True)
    assert len(r["pairs"]) == 2 and r["pairs"].set_index("new_url").loc["https://staging.new.com/a", "ratio"] == 0.5
    r2 = ipr.compare(pre, post, None, ignore_host=False)
    assert len(r2["pairs"]) == 0 and len(r2["unmapped_old"]) == 2


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(ROOT, "ipr_cli.py"), "compare",
           "--pre-inlinks", f"{SAMPLE}/pre_all_inlinks.csv", "--pre-internal", f"{SAMPLE}/pre_internal_all.csv",
           "--post-inlinks", f"{SAMPLE}/post_all_inlinks.csv", "--post-internal", f"{SAMPLE}/post_internal_all.csv",
           "--map", f"{SAMPLE}/redirect_map.csv", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    for f in ("pairs.csv", "losers.csv", "unmapped_old.csv", "orphans_post.csv", "summary.csv", "segments.csv"):
        assert (out / f).exists()
    out2 = tmp_path / "out2"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "ipr_cli.py"), "scenario", "--inlinks", f"{SAMPLE}/post_all_inlinks.csv",
                        "--internal", f"{SAMPLE}/post_internal_all.csv", "--changes", f"{SAMPLE}/scenario_changes.csv",
                        "--out", str(out2)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    t = pd.read_csv(out2 / "scenario_targets.csv")
    assert t.delta.iloc[0] > 0
    out3 = tmp_path / "out3"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "ipr_cli.py"), "score", "--inlinks", f"{SAMPLE}/post_all_inlinks.csv",
                        "--internal", f"{SAMPLE}/post_internal_all.csv", "--position-weights", "content=1,navigation=.5,footer=.2",
                        "--out", str(out3)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


def test_streamlit_app_runs_with_sample_data():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.sidebar.checkbox(key="use_sample").check()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) >= 10                                         # score, compare and plan tabs all rendered


def test_raw_pagerank_is_a_probability_and_pr_x_avg_is_relative():
    g = ipr.build_graph(sample("post_all_inlinks.csv"), sample("post_internal_all.csv"))
    s = ipr.score_graph(g)
    assert 0 < s.pr.max() < 1 and abs(s.pr.sum() - 1) < 1e-9          # true PageRank can never exceed 1
    assert abs(s.pr_x_avg.mean() - 1) < 1e-9                          # 1.0 = average page
    assert (s.pr_x_avg > 1).any()                                     # above-average pages exceed 1.0, by design


# ----------------------------------------------------------------------------- related-link suggestions
import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

import ipr_related as rel  # noqa: E402

ACC = "https://www.example-shop.com/shop/accessories/"


def _related_setup():
    inl, itn = sample("post_all_inlinks.csv"), sample("post_internal_all.csv")
    g = ipr.build_graph(inl, itn)
    sc = ipr.score_graph(g)
    emb = rel.load_embeddings(rel.read_embeddings_csv(f"{SAMPLE}/post_embeddings.csv"))
    pages = rel.build_page_text(ipr.read_sf_csv(f"{SAMPLE}/post_internal_all.csv", True),
                                ipr.read_sf_csv(f"{SAMPLE}/post_page_text.csv", True))
    tix = rel.build_text_index(pages)
    anchors = rel.anchor_counts(rel.read_anchor_table(f"{SAMPLE}/post_all_inlinks.csv"))
    return g, sc, emb, tix, anchors


def test_load_embeddings_accepts_wide_and_packed_and_skips_bad_rows():
    wide = pd.DataFrame({"Address": ["https://a.com/x", "https://a.com/y", "https://a.com/x", "https://a.com/z"],
                         "dim_0": [1.0, 0.0, 5.0, 0.0], "dim_1": [0.0, 2.0, 5.0, 0.0], "Title 1": ["t"] * 4})
    e = rel.load_embeddings(wide)
    assert e.urls == ["https://a.com/x", "https://a.com/y"] and e.dims == 2     # dup and zero vector dropped, title ignored
    assert e.dropped == 2 and np.allclose(np.linalg.norm(e.matrix, axis=1), 1)
    packed = pd.DataFrame({"URL": ["https://a.com/x", "https://a.com/y", "https://a.com/bad"],
                           "embedding": ["[1, 0, 0]", "0.5,0.5,0", "[1, 2]"]})
    p = rel.load_embeddings(packed)
    assert p.dims == 3 and len(p.urls) == 2 and p.dropped == 1                   # wrong-length vector dropped
    with pytest.raises(ValueError):
        rel.load_embeddings(pd.DataFrame({"Address": ["https://a.com/x"]}))


def test_related_suggestions_pick_relevant_pages_with_natural_anchors():
    g, sc, emb, tix, anchors = _related_setup()
    r = rel.suggest_related_links(g, sc, [ACC], emb, text_index=tix, anchors=anchors, k=5)
    assert len(r) == 5
    existing = {g.nodes[i] for i in g.src[g.dst == g.idx[ACC]]}
    assert not set(r.source) & existing and ACC not in set(r.source)
    # the 5 most related pages that can still link are accessories articles
    assert all(int(u.rsplit("-", 1)[1]) % 3 == 2 for u in r.source), list(r.source)
    assert (r.suggested_anchor.str.lower() == "leather accessories").all()
    assert (r.anchor_basis == "Phrase already on the linking page").all()
    assert all("leather accessories" in c.lower() for c in r.context)            # the sentence to put the link in
    assert r.anchor_note.str.contains("Vary it").all()                           # breadcrumbs already use this anchor
    assert r.similarity.is_monotonic_decreasing                                  # ordered by priority; similar weights here


def test_related_suggestions_text_fallback_and_options():
    g, sc, emb, tix, anchors = _related_setup()
    r = rel.suggest_related_links(g, sc, [ACC], tix, text_index=tix, k=10)
    posts = [u for u in r.source if "/resources/post-" in u and int(u.rsplit("-", 1)[1]) % 3 == 2]
    assert len(posts) >= 4 and (r.method == tix.method).all()
    strict = rel.suggest_related_links(g, sc, [ACC], emb, text_index=tix, k=50, min_percentile=95)
    loose = rel.suggest_related_links(g, sc, [ACC], emb, text_index=tix, k=50, min_percentile=50)
    assert 0 < len(strict) < len(loose)
    assert strict.similarity.min() >= loose.similarity.sort_values().iloc[len(loose) - len(strict)] - 1e-9
    pure = rel.suggest_related_links(g, sc, [ACC], emb, k=20, equity_weight=0.0)
    assert pure.priority.is_monotonic_decreasing and np.allclose(pure.priority, pure.similarity)
    floor = rel.suggest_related_links(g, sc, [ACC], emb, k=20, min_percentile=0, min_similarity=0.7)
    assert len(floor) and (floor.similarity >= 0.7).all()
    miss = rel.suggest_related_links(g, sc, ["https://www.example-shop.com/nope"], emb)
    assert miss.empty and miss.attrs["skipped"] == ["https://www.example-shop.com/nope"]
    few = rel.suggest_related_links(g, sc, [ACC], emb, k=50, min_percentile=0, max_source_outlinks=10)
    assert len(few) and (few.source_outlinks <= 10).all()


def test_anchor_falls_back_when_phrase_not_on_source_page():
    g, sc, emb, tix, anchors = _related_setup()
    a = rel.suggest_anchor(tix, "https://www.example-shop.com/shop/shoes/", ACC)      # shoes category never says it
    assert a["suggested_anchor"].lower() == "leather accessories" and "by hand" in a["anchor_basis"]
    assert rel.suggest_anchor(None, "x", "y")["suggested_anchor"] == ""


def test_related_plan_feeds_the_scenario_engine():
    g, sc, emb, tix, anchors = _related_setup()
    r = rel.suggest_related_links(g, sc, [ACC], emb, text_index=tix, k=5)
    scen = ipr.score_graph(ipr.apply_scenario(g, r), 0.85)
    rep = ipr.scenario_report(sc, scen, [ACC])
    assert rep["summary"]["target_pr_x_avg_gain"] > 0


def test_embed_pages_against_a_fake_ollama(tmp_path):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["model"], seen["first"] = body["model"], body["input"][0]
            out = json.dumps({"embeddings": [[float(len(t) % 7), 1.0, 2.0] for t in body["input"]]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "emb.csv"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "embed_pages.py"), "--internal", f"{SAMPLE}/post_internal_all.csv",
                        "--page-text", f"{SAMPLE}/post_page_text.csv", "--out", str(out),
                        "--host", f"http://127.0.0.1:{srv.server_port}", "--batch", "10"],
                       capture_output=True, text=True, cwd=ROOT)
    srv.shutdown()
    assert r.returncode == 0, r.stderr
    e = rel.load_embeddings(rel.read_embeddings_csv(str(out)))
    assert e.dims == 3 and len(e.urls) == 152                                    # live pages only
    assert seen["model"] == "embeddinggemma" and seen["first"].startswith("title: ")


def test_cli_related_suggestions(tmp_path):
    t = tmp_path / "targets.txt"
    t.write_text(ACC + "\n")
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "ipr_cli.py"), "scenario", "--inlinks", f"{SAMPLE}/post_all_inlinks.csv",
                        "--internal", f"{SAMPLE}/post_internal_all.csv", "--suggest-for", str(t),
                        "--embeddings", f"{SAMPLE}/post_embeddings.csv", "--page-text", f"{SAMPLE}/post_page_text.csv",
                        "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    ch = pd.read_csv(out / "scenario_changes.csv")
    assert len(ch) == 5 and "suggested_anchor" in ch.columns and (pd.read_csv(out / "scenario_targets.csv").delta > 0).all()
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "ipr_cli.py"), "scenario", "--inlinks", f"{SAMPLE}/post_all_inlinks.csv",
                         "--internal", f"{SAMPLE}/post_internal_all.csv", "--suggest-for", str(t), "--strength-only",
                         "--out", str(tmp_path / "o2")], capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0, r2.stderr
