"""Repeat the VizieR requests whose replies hold no table: the last of the four fetches behind the test of the
direction of a new layer (layer_direction_test.py, Paper II section 3.1).

On 2026-11-22 the VizieR mirror at the Harvard-Smithsonian Center for Astrophysics answered some requests of
fetch_vizier.py and fetch_vizier_hi.py with a database connection error or with an empty resource, and a request
repeated by hand then returned the catalog's row. This script finds every reply listed in vizier-manifest.tsv and
vizier-hi-manifest.tsv that holds no "#Table" line, repeats its request with the same address up to three times,
ten seconds apart, and saves every repeated reply as fetched under vizier/<name>__retry<k>.tsv in the data folder.
The failed replies are kept. vizier-retry-manifest.tsv in the data folder lists the repeated requests with their
address, file, bytes, SHA-256, time of fetch and the reply each repeats. The data folder is fetched by the reader
with the four fetch scripts, and its SHA256SUMS checks a new fetch against the replies of 2026-11-22 that Paper II
used.

Run from the repository root after fetch_vizier_hi.py (about 13 minutes on 2026-11-22, 76 requests; standard library
only):
    python rotation-curves/fetch_retry.py data/sparc-velocities-2026-11-22
"""
import hashlib
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_vizier import AGENT  # noqa: E402


def failed(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    return "#Table" not in txt


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    out_man = os.path.join(dest, "vizier-retry-manifest.tsv")
    if os.path.exists(out_man):
        sys.exit("vizier-retry-manifest.tsv exists; a later repetition goes into a new dated folder")
    rows = ["galaxy\tcatalog\turl\tfile\tbytes\tsha256\tfetched_utc\tretry_of"]
    for man in ("vizier-manifest.tsv", "vizier-hi-manifest.tsv"):
        p = os.path.join(dest, man)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8").read().splitlines()[1:]:
            f = line.split("\t")
            gal, cat, url, rel = f[0], f[1], f[2], f[3]
            if not failed(os.path.join(dest, rel)):
                continue
            stem = rel[:-4]
            for k in range(1, 4):
                req = urllib.request.Request(url, headers={"User-Agent": AGENT})
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        body = r.read()
                except Exception as e:
                    body = ("# request failed: " + str(e) + "\n").encode("utf-8")
                fn = f"{stem}__retry{k}.tsv"
                with open(os.path.join(dest, fn), "wb") as fh:
                    fh.write(body)
                stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                rows.append("\t".join([gal, cat, url, fn, str(len(body)), hashlib.sha256(body).hexdigest(), stamp, rel]))
                ok = b"#Table" in body
                print(f"{gal:12s} {cat:22s} retry {k}: {len(body):6d} bytes, {'table' if ok else 'no table'}", flush=True)
                time.sleep(10.0)
                if ok:
                    break
    with open(out_man, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
