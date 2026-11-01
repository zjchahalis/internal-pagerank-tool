# Internal PageRank Comparison Tool

Score how link equity flows through your own site, compare it **before and after a migration**, and **test an internal linking plan** before it ships. Built to run on Screaming Frog exports.

Three ways to use it, same engine:

| | |
|---|---|
| `app.py` | Streamlit app. Upload CSVs, get tables, charts and a zip of reports. Deployable for free. |
| `ipr_cli.py` | Command line. Good for repeatable runs and CI. |
| `ipr_core.py` | The engine. Import it in a notebook. |
| `ipr_related.py` | Related-link suggestions (embeddings or TF-IDF) with anchor text. |
| `embed_pages.py` | Makes the embeddings CSV with a local Ollama model such as EmbeddingGemma. |

## Quick start

```bash
pip install -r requirements.txt
python make_sample.py                 # writes made-up before/after crawls to sample_data/
streamlit run app.py                  # tick "Use the bundled sample data" in the sidebar
```

Or from the command line:

```bash
python ipr_cli.py compare \
  --pre-inlinks pre_all_inlinks.csv   --pre-internal pre_internal_all.csv \
  --post-inlinks post_all_inlinks.csv --post-internal post_internal_all.csv \
  --map redirect_map.csv --out out/
```

## What to export from Screaming Frog

Export two CSVs from **each** crawl (before and after).

1. **All Inlinks**: Bulk Export > Links > All Inlinks. This is the link graph. Uses `Source`, `Destination`, `Type`, `Follow`, `Link Position`, `Status Code`. (All Outlinks also works.)
2. **Internal: All**: Internal tab, filter All, Export. Uses `Address`, `Status Code`, `Content Type`, `Indexability`, `Redirect URL`, `Crawl Depth`, `Link Score`.

The Internal export is optional but recommended. It lets the tool drop non-HTML files, pass equity through redirects, ignore non-indexable pages in reports, and flag orphans.

Crawl tips: turn on JavaScript rendering if links are built in the browser; keep scope identical before and after; enable linked-sitemap crawling so unlinked pages show up as orphans.

For a migration, also bring a **redirect map** CSV with `Source` and `Target` (also accepts Old/New or From/To). The embeddings mapping from the talk produces exactly this.

## Plain-English headings and tooltips

The app shows friendly names and a hover tooltip on every table heading and metric, and has a **Glossary** tab with all of them. The mapping lives in `glossary.py`, so you can reword anything without touching the maths. The command line and CSV output from `ipr_cli.py` keep the technical names used below; zip downloads from the app use the friendly names.

| In the app | Technical name | Meaning |
|---|---|---|
| Link strength (× average) | `pr_x_avg` | 1.0 = average page, 3.0 = three times average |
| PageRank (0 to 1) | `pr` | True PageRank, sums to 1 across the site |
| Link score (0 to 100) | `score` | Log scale of strength |
| Rank (percentile) | `percentile` | 90 = better linked than 90% of pages |
| Links in / Links out | `inlinks` / `outlinks` | Count of links |
| Clicks from home | `depth` | Fewest clicks from the home page |
| No links in | `orphan` | Nothing links to the page |
| Cut off from home | `unreachable` | No click path from the home page |
| Kept (after ÷ before) | `ratio` | 0.5 = lost half |

## What it reports

**Per page** (`scores.csv`)

- `pr`: true PageRank. Always between 0 and 1 and sums to 1 across the site, so each page holds a tiny share that shrinks as the site grows.
- `pr_x_avg`: PageRank as a **multiple of the average page** (`pr` times page count). **1.0 is average, 3.0 is three times average.** Values above 1.0 are normal. Use this to compare crawls, because raw `pr` is not comparable between sites of different size.
- `score`: 0 to 100 log scale of PageRank (similar in spirit to Screaming Frog's Link Score, which is also carried through as `sf_link_score` for a sanity check).
- `percentile`, `inlinks`, `outlinks`, `depth` (clicks from home), `orphan` (no links in), `unreachable` (no followed path from home, which also catches sections that only link to themselves).

**Before versus after** (`compare`)

- `losers` / `gainers`: pages whose `pr_x_avg` fell below (or rose above) a threshold, sorted by absolute equity lost so important pages come first.
- `unmapped_old`: old pages that carried equity but have no destination in the new crawl.
- `deeper`: pages pushed deeper or cut off from the home page.
- `orphans_post` and `new_pages`: pages with no links in, and pages with no old equivalent.
- `segments`: share of total PageRank by section, before and after.
- `summary`: how many of the top 100 pages stayed in the top 100, rank stability, share of old equity with no destination.

When several old URLs redirect to one new URL their `pr_x_avg` is **summed** before comparing, so a consolidation is judged against the equity it should inherit.

**Linking plans** (`scenario`)

Pick the pages you want to lift and the tool suggests which pages should link to them, then applies the links and shows who gains and who loses link strength. You can also bring your own changes file (`action, source, destination, weight`).

Suggestions are built on **relevance first**:

1. **Match by meaning.** Upload a CSV of page embeddings (below). Related pages have a high cosine similarity.
2. **No embeddings? Match by words.** The fallback is TF-IDF over single words and two-word phrases from each page's title, H1, meta description and, if you upload it, body text. No model, no setup, works on a hosted copy.
3. **Eligibility.** A page can be suggested only if it is indexable, does not already link to the target, and is among the most similar pages that could still link (default: top 10%, adjustable).
4. **Order.** Among eligible pages, `priority = similarity x ((1 - w) + w x link-value percentile)`, where link value is the page's strength divided by its links out and `w` defaults to 0.5. Set `w` to 0 to order by relevance alone. A strong page that is unrelated to the target is never suggested.
5. **Link text.** The tool looks for the target's H1, its title, or its most distinctive phrases inside the linking page's body and suggests that exact wording, with the sentence it sits in. If no natural phrase exists it falls back to the target's H1 and marks it "place by hand". If the suggested wording already makes up 30% or more of the links to the target (read from the Anchor column of All Inlinks), it warns you to vary it.

The old strength-only suggestion is still available (`Suggest by link strength only`, or `--strength-only` on the command line), but it ignores meaning.

**Making the embeddings file**

```bash
ollama pull embeddinggemma
python embed_pages.py --internal internal_all.csv --page-text page_text.csv --out embeddings.csv
```

`embed_pages.py` sends text only to the Ollama server on your machine. The file it writes has `Address` plus one numeric column per dimension. The app also accepts a file with a single column holding the whole vector (for example `[0.1, 0.2, ...]`), so exports from other tools work. For body-copy anchors, export a CSV of `Address` plus the main content text (a Screaming Frog Custom Extraction works). Embeddings from your own pipeline are fine as long as every vector comes from the same model.

Limits: the tool cannot see the page's HTML, so it cannot tell whether a suggested phrase is already linked elsewhere in the copy, and a person should review every suggestion. Similarity scores are not comparable across models, so use the "most related pages" setting rather than a fixed number unless you know your model's range.

## Using it in a replatforming

1. **Baseline.** Crawl production before anything changes. Score it.
2. **Plan.** Decide which templates and pages must hold or gain equity (money pages, hubs). Use *Test a linking plan* to try navigation and module changes on the baseline.
3. **Staging check.** Crawl staging. Compare to the baseline with your redirect map (or match by path if the hostname is the only difference). Fix losers, orphans and pages pushed deeper **before launch**.
4. **Post-launch.** Crawl production again and compare to the same baseline. Check `unmapped_old` and `orphans_post` first.
5. **Segments.** Define sections with regexes (for example `Products=^/products/|^/shop/[^/]+/[^/]+$`) so renamed directories line up before and after.

## Method and limits

- This is a model of equity flow through *your* links. It is not Google's PageRank and does not predict rankings. Use it to compare two states of the same site and to find what changed.
- Defaults: damping 0.85, nofollow links ignored, redirects resolved to their final target, links to 4xx/5xx pages reported but not counted, duplicate links between the same two pages counted once.
- Position weighting is an assumption. The built-in preset (content 1, navigation and header 0.5, aside 0.3, footer 0.2) is a starting point, not a measurement.
- Boilerplate links (nav, footer) can dominate a graph. Try the *Ignore links in these positions* option and compare.
- Pre and post `pr_x_avg` are comparable because they are normalized by page count, but big changes in site size or structure still move every page. Read `percentile` and `ratio` together.
- CSV only. Very large crawls (millions of link rows) are better run locally than on a hosted free tier.

## Deploy it free

**Streamlit Community Cloud**

1. Put this folder in a GitHub repository.
2. At share.streamlit.io, create an app from the repo with `app.py` as the entry file.
3. To show it on your own site, embed the app URL with `?embed=true` in an iframe.

Hugging Face Spaces is a free alternative. Check current free-tier limits (memory, private apps) before relying on either.

**Privacy.** On a hosted copy, uploaded crawls go to the host's servers. Client crawls are confidential: run the app locally for client data, and use the hosted copy for the sample data or your own site. The app does not write uploads to disk or store them, but the host still receives them.

## Tests

```bash
python -m pytest tests -q
```

The suite checks related-link suggestions (embeddings and text, anchors, options), the embedding helper against a fake Ollama server, PageRank against networkx (plain and weighted), redirect resolution and filtering, depth and orphan logic, scenarios, link suggestions, the sample migration story (merged pages, orphaned section, buried hub, removed pages), the CLI, and the Streamlit app.
