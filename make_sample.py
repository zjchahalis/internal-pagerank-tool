"""Generate a small synthetic site in Screaming Frog export format, before and after a replatform.

    python make_sample.py            # writes ./sample_data/*.csv

The data is made up. It exists so you can try the tool and so the tests have known answers.
What changes in the "migration":
  * product pages lose their "related products" links
  * the blog moves to /resources/ and is linked from a single product page (it gets much deeper)
  * /shop/accessories/ is published but nothing links to it (orphan section)
  * three old blog posts are merged into one guide (many-to-one redirect)
  * five old posts are removed and left out of the map (equity with no destination)
"""
from __future__ import annotations

import csv
import os
import random
from collections import deque

import re
import zlib

import numpy as np

OLD = "https://www.example-shop.com"
CATS = ["shoes", "bags", "accessories"]
N_PROD, N_POSTS = 40, 30


PHRASES = {
    "shoes": ["running shoes", "leather boots", "trail sneakers", "waterproof hiking boots", "canvas sandals"],
    "bags": ["leather backpack", "weekend duffel bag", "laptop messenger bag", "canvas tote bag", "travel pouch"],
    "accessories": ["leather belts", "wool scarves", "polarized sunglasses", "leather wallets", "silk ties"],
}
CATPH = {"shoes": "shoes and boots", "bags": "bags and backpacks", "accessories": "leather accessories"}
CATTITLE = {"shoes": "Shoes and boots", "bags": "Bags and backpacks", "accessories": "Leather accessories"}


def content_for(url: str) -> dict:
    """Title, H1, meta description, body text and topic for a page, from its URL pattern (old or new site)."""
    path = url.replace("https://www.example-shop.com", "") or "/"
    rnd = random.Random(path)                                    # own RNG: link generation is unaffected
    m = re.match(r"^/(?:products/|shop/[a-z]+/)(shoes|bags|accessories)-(\d+)$", path)
    if m:
        c, i = m.group(1), int(m.group(2))
        item = PHRASES[c][i % 5]
        name = f"{item.capitalize()} {i}"
        body = (f"{name} is made for everyday wear and built to last. It belongs to our range of {CATPH[c]}. "
                f"Pair it with {PHRASES[c][(i + 1) % 5]} for a complete look. "
                f"Customers who like {item} often browse {PHRASES[c][(i + 2) % 5]} as well. "
                f"Free returns on all {CATPH[c]}.")
        return dict(topic=c, title=f"{name} | Example Shop", h1=name, meta=f"Shop {item} in our {CATPH[c]} collection.", body=body)
    m = re.match(r"^/(?:shop/)?(shoes|bags|accessories)/$", path)
    if m:
        c = m.group(1)
        its = ", ".join(PHRASES[c][:4])
        return dict(topic=c, title=f"{CATTITLE[c]} | Example Shop", h1=CATTITLE[c],
                    meta=f"Browse {its} and more.",
                    body=f"Our full collection of {CATPH[c]}. Browse {its} and more. Free returns on every order.")
    m = re.match(r"^/(?:blog|resources)/post-(\d+)$", path)
    if m:
        i = int(m.group(1))
        c = CATS[i % 3]
        a, b = PHRASES[c][i % 5], PHRASES[c][(i + 3) % 5]
        extra = rnd.choice([x for x in CATS if x != c])
        body = (f"How to choose {a}. In this guide we cover {a} and {b}, and how to look after them. "
                f"If you are building a wardrobe of {CATPH[c]}, start with one versatile piece and add from there. "
                f"We also compare prices and materials across our {CATPH[c]} range. "
                f"A quick word on {PHRASES[extra][0]}, which pair well with most outfits.")
        return dict(topic=c, title=f"How to choose {a} | Example Shop Journal", h1=f"How to choose {a}",
                    meta=f"A buyer's guide to {a} and {b}.", body=body)
    if path.endswith("migration-guide"):
        return dict(topic="site", title="Our website migration guide | Example Shop", h1="Our website migration guide",
                    meta="What changed when we moved to a new website.",
                    body="We moved to a new website platform. This guide explains what changed, how old addresses redirect, "
                         "and how to find your favourite pages again.")
    if path in ("/blog/", "/resources/"):
        return dict(topic="journal", title="Journal | Example Shop", h1="Journal", meta="Guides and advice.",
                    body="Guides and advice on shoes, bags and accessories from the Example Shop team.")
    if path == "/shop/new-arrivals/":
        return dict(topic="arrivals", title="New arrivals | Example Shop", h1="New arrivals", meta="The latest shoes and bags.",
                    body="The newest shoes and boots and bags and backpacks, fresh in this week.")
    if path == "/":
        return dict(topic="home", title="Example Shop | Shoes, bags and accessories", h1="Example Shop",
                    meta="Shoes, bags and leather accessories.",
                    body="Shoes and boots, bags and backpacks, and leather accessories for every day.")
    return dict(topic="info", title=f"{path.strip('/').capitalize() or 'Page'} | Example Shop",
                h1=path.strip("/").capitalize() or "Page", meta="Information.", body=f"Information about {path.strip('/')}.")


def embedding_for(topic: str, rng_seed: str, dims: int = 32) -> np.ndarray:
    """Made-up embedding: a shared vector per topic plus page-level noise (stands in for EmbeddingGemma output)."""
    topics = ["shoes", "bags", "accessories", "site", "journal", "info"]
    cent = np.random.default_rng(123).normal(size=(len(topics), dims))
    own = np.random.default_rng(zlib.crc32(rng_seed.encode()))
    noise = own.normal(size=dims) * 0.45
    if topic in topics:
        v = cent[topics.index(topic)] + noise
    elif topic == "arrivals":
        v = 0.7 * cent[0] + 0.7 * cent[1] + noise
    else:                                                        # home
        v = (cent[0] + cent[1] + cent[2]) / 3 + noise
    return v


def old_site():
    pages, links = {}, []
    html = lambda: dict(status=200, ctype="text/html; charset=utf-8", index="Indexable", redirect="")
    home = OLD + "/"
    pages[home] = html()
    static = [OLD + "/about", OLD + "/contact", OLD + "/returns"]
    cat_url = {c: f"{OLD}/{c}/" for c in CATS}
    prod = {c: [f"{OLD}/products/{c}-{i}" for i in range(1, N_PROD + 1)] for c in CATS}
    posts = [f"{OLD}/blog/post-{i}" for i in range(1, N_POSTS + 1)]
    blog_idx = OLD + "/blog/"
    for u in static + list(cat_url.values()) + [blog_idx] + posts + [p for c in CATS for p in prod[c]]:
        pages[u] = html()
    sale, promo = OLD + "/sale", OLD + "/old-promo"
    pages[sale] = dict(status=301, ctype="text/html; charset=utf-8", index="Non-Indexable", redirect=cat_url["shoes"])
    pages[promo] = dict(status=404, ctype="text/html; charset=utf-8", index="Non-Indexable", redirect="")
    rnd = random.Random(7)

    for s in list(pages):
        if pages[s]["status"] != 200:
            continue
        for t in [home] + list(cat_url.values()) + [static[0], static[1], sale]:
            if t != s:
                links.append((s, t, "Navigation", "true"))
        for t in [static[2], static[1], blog_idx]:
            if t != s:
                links.append((s, t, "Footer", "true"))
        links.append((s, OLD + "/login", "Navigation", "false"))

    for c in CATS:
        for p in rnd.sample(prod[c], 4):
            links.append((home, p, "Content", "true"))
        links.append((home, cat_url[c], "Content", "true"))
        for p in prod[c]:
            links.append((cat_url[c], p, "Content", "true"))
            links.append((p, cat_url[c], "Navigation", "true"))
            for q in rnd.sample([x for x in prod[c] if x != p], 4):
                links.append((p, q, "Content", "true"))
    for po in posts:
        links.append((blog_idx, po, "Content", "true"))
        links.append((po, blog_idx, "Content", "true"))
        for c in rnd.sample(CATS, 2):
            links.append((po, rnd.choice(prod[c]), "Content", "true"))
    links.append((home, promo, "Content", "true"))
    return pages, links, dict(home=home, cat=cat_url, prod=prod, posts=posts, blog=blog_idx, static=static, sale=sale)


def new_site(o):
    N = "https://www.example-shop.com"
    pages, links = {}, []
    html = lambda: dict(status=200, ctype="text/html; charset=utf-8", index="Indexable", redirect="")
    home = N + "/"
    m = {}                                                   # old -> new
    m[o["home"]] = home
    m[o["static"][0]], m[o["static"][1]], m[o["static"][2]] = N + "/about", N + "/contact", N + "/returns"
    for c in CATS:
        m[o["cat"][c]] = f"{N}/shop/{c}/"
        for p in o["prod"][c]:
            m[p] = f"{N}/shop/{c}/{p.rsplit('/', 1)[1]}"
    m[o["blog"]] = N + "/resources/"
    guide = N + "/resources/migration-guide"
    for i, p in enumerate(o["posts"], 1):
        if i <= 3:
            m[p] = guide                                     # many old posts -> one guide
        elif i <= 8:
            pass                                             # removed, not in map
        else:
            m[p] = f"{N}/resources/post-{i}"
    m[o["sale"]] = f"{N}/shop/shoes/"

    for u in set(m.values()):
        pages[u] = html()
    new_arrivals = N + "/shop/new-arrivals/"
    pages[new_arrivals] = html()
    rnd = random.Random(11)

    nav = [home] + [f"{N}/shop/{c}/" for c in CATS if c != "accessories"] + [N + "/about"]
    foot = [N + "/returns", N + "/contact"]
    for s in pages:
        for t in nav:
            if t != s:
                links.append((s, t, "Navigation", "true"))
        for t in foot:
            if t != s:
                links.append((s, t, "Footer", "true"))
    for c in CATS:
        cu = f"{N}/shop/{c}/"
        prods = [m[p] for p in o["prod"][c]]
        if c != "accessories":
            for p in rnd.sample(prods, 6):
                links.append((home, p, "Content", "true"))
        for p in prods:
            links.append((cu, p, "Content", "true"))
            links.append((p, cu, "Navigation", "true"))           # breadcrumb only: no related links
    links.append((home, new_arrivals, "Content", "true"))
    for c in ["shoes", "bags"]:
        for p in rnd.sample([m[x] for x in o["prod"][c]], 5):
            links.append((new_arrivals, p, "Content", "true"))
    res = N + "/resources/"
    links.append((m[o["prod"]["shoes"][0]], res, "Content", "true"))   # blog hub linked from one deep page only
    for po in o["posts"]:
        if po in m:
            t = m[po]
            links.append((res, t, "Content", "true"))
            links.append((t, res, "Content", "true"))
            links.append((t, rnd.choice(list(m[x] for x in o["prod"]["shoes"])), "Content", "true"))
    return pages, links, m


def crawl_depths(pages, links, home):
    adj = {}
    for s, t, *_ in links:
        if pages.get(s, {}).get("status") == 200 and pages.get(t, {}).get("status") == 200:
            adj.setdefault(s, set()).add(t)
    d, q = {home: 0}, deque([home])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def write(prefix, pages, links, home, outdir):
    # redirects appear to point their source at the target when links go through them
    with open(os.path.join(outdir, f"{prefix}_all_inlinks.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Type", "Source", "Destination", "Anchor", "Status Code", "Status", "Follow",
                    "Link Position", "Link Origin"])
        for s, t, pos, fol in links:
            info = pages.get(t, {"status": 200 if not t.endswith("/login") else 200})
            anchor = "Log in" if t.endswith("/login") else content_for(t)["h1"] if info["status"] == 200 else ""
            w.writerow(["Hyperlink", s, t, anchor, info["status"], "OK" if info["status"] == 200 else "Redirect"
                        if info["status"] == 301 else "Client Error", fol, pos, "HTML"])
    depth = crawl_depths(pages, links, home)
    inl = {}
    for s, t, pos, fol in links:
        inl[t] = inl.get(t, 0) + 1
    with open(os.path.join(outdir, f"{prefix}_internal_all.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Address", "Content Type", "Status Code", "Status", "Indexability", "Indexability Status",
                    "Crawl Depth", "Link Score", "Redirect URL", "Inlinks", "Title 1", "H1-1", "Meta Description 1"])
        for u, i in pages.items():
            c = content_for(u) if i["status"] == 200 else dict(title="", h1="", meta="")
            w.writerow([u, i["ctype"], i["status"], {200: "OK", 301: "Moved Permanently", 404: "Not Found"}[i["status"]],
                        i["index"], "" if i["index"] == "Indexable" else "Redirected" if i["status"] == 301 else "Client Error",
                        depth.get(u, ""), "", i["redirect"], inl.get(u, 0), c["title"], c["h1"], c["meta"]])


def main(outdir="sample_data"):
    os.makedirs(outdir, exist_ok=True)
    pages, links, o = old_site()
    write("pre", pages, links, o["home"], outdir)
    npages, nlinks, m = new_site(o)
    write("post", npages, nlinks, "https://www.example-shop.com/", outdir)
    with open(os.path.join(outdir, "redirect_map.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Target"])
        for k, v in m.items():
            w.writerow([k, v])
    # proposed linking plan to test on the post crawl: link the orphaned accessories section
    N = "https://www.example-shop.com"
    with open(os.path.join(outdir, "scenario_changes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "source", "destination", "weight"])
        w.writerow(["add", N + "/", N + "/shop/accessories/", 1])
        w.writerow(["add", N + "/shop/new-arrivals/", N + "/shop/accessories/", 1])
        w.writerow(["add", N + "/shop/bags/", N + "/shop/accessories/", 1])
    # page text (what a custom extraction of the body copy would give) and embeddings for the new site
    live = [u for u, i in npages.items() if i["status"] == 200]
    with open(os.path.join(outdir, "post_page_text.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Address", "Text"])
        for u in live:
            w.writerow([u, content_for(u)["body"]])
    dims = 32
    with open(os.path.join(outdir, "post_embeddings.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Address"] + [f"dim_{j}" for j in range(dims)])
        for u in live:
            v = embedding_for(content_for(u)["topic"], u, dims)
            w.writerow([u] + [f"{x:.5f}" for x in v])
    with open(os.path.join(outdir, "segments.txt"), "w", encoding="utf-8") as f:
        f.write("Products=^/products/|^/shop/[^/]+/[^/]+$\nCategories=^/(shoes|bags|accessories)/$|^/shop/[^/]+/$\n"
                "Blog and resources=^/(blog|resources)/\n")
    print(f"Wrote sample data to {outdir}/ ({len(pages)} old pages, {len(npages)} new pages)")


if __name__ == "__main__":
    main()
