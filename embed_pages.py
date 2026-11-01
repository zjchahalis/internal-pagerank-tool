#!/usr/bin/env python3
"""Make an embeddings CSV for the linking-plan tab using a local Ollama model (default: embeddinggemma).

    ollama pull embeddinggemma
    python embed_pages.py --internal internal_all.csv --page-text page_text.csv --out embeddings.csv

Input: the Screaming Frog Internal: All export (uses Title 1, H1-1, Meta Description 1) and/or a CSV with
Address plus a body text column (for example a Custom Extraction of the main content). Output: Address plus one
numeric column per dimension, which the app and ipr_cli.py read directly. Nothing leaves your machine: the
text goes only to the Ollama server you point --host at.

Uses only the Python standard library plus pandas.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import ipr_core as ipr
import ipr_related as rel


def _post(host: str, path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(host.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed_batch(host: str, model: str, texts: list[str]) -> list[list[float]]:
    try:
        return _post(host, "/api/embed", {"model": model, "input": texts, "truncate": True})["embeddings"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    # older Ollama: one text per call
    return [_post(host, "/api/embeddings", {"model": model, "prompt": t})["embedding"] for t in texts]


def document_text(row, max_chars: int, prefix: bool) -> str:
    body = " ".join(x for x in (row.h1, row.meta, row.body) if x).strip()[:max_chars]
    if prefix:                                    # EmbeddingGemma's document format; check the model card for other models
        return f"title: {row.title.strip() or 'none'} | text: {body}"
    return " ".join(x for x in (row.title, body) if x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internal", help="Screaming Frog Internal: All CSV")
    ap.add_argument("--page-text", help="CSV with Address and a body text column")
    ap.add_argument("--out", default="embeddings.csv")
    ap.add_argument("--model", default="embeddinggemma")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=6000, help="characters of text per page sent to the model")
    ap.add_argument("--no-prefix", action="store_true", help="do not use EmbeddingGemma's 'title: ... | text: ...' format")
    ap.add_argument("--dims", type=int, help="keep only the first N dimensions (for models trained to be truncated)")
    a = ap.parse_args()
    if not (a.internal or a.page_text):
        sys.exit("Give --internal and/or --page-text")
    internal = ipr.read_sf_csv(a.internal, usecols_all=True) if a.internal else None
    text = ipr.read_sf_csv(a.page_text, usecols_all=True) if a.page_text else None
    pages = rel.build_page_text(internal, text)
    if internal is not None:                      # only embed live HTML pages
        sc, ic = ipr._col(internal, "Status Code"), ipr._col(internal, "Address")
        ct = ipr._col(internal, "Content Type")
        ok = internal[ic].astype(str).map(ipr.norm_url)
        mask = np.ones(len(internal), dtype=bool)
        if sc:
            mask &= internal[sc].astype(str).str.startswith("2").to_numpy()
        if ct:
            mask &= internal[ct].astype(str).str.contains("html", case=False).to_numpy()
        pages = pages[pages.url.isin(set(ok[mask]))]
    rows = list(pages.itertuples())
    if not rows:
        sys.exit("No pages with text were found.")
    print(f"Embedding {len(rows)} pages with {a.model} at {a.host}")
    t0, written = time.time(), 0
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            vecs = embed_batch(a.host, a.model, [document_text(r, a.max_chars, not a.no_prefix) for r in chunk])
            for r, v in zip(chunk, vecs):
                v = v[:a.dims] if a.dims else v
                if written == 0:
                    w.writerow(["Address"] + [f"dim_{j}" for j in range(len(v))])
                w.writerow([r.url] + [f"{x:.6f}" for x in v])
                written += 1
            print(f"  {written}/{len(rows)}  ({time.time() - t0:.0f}s)", end="\r")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
