"""Fetch the homogenized HI velocity of every SPARC galaxy from Paturel et al. (2003), VizieR catalog VII/238: the third
of the four fetches behind the test of the direction of a new layer (layer_direction_test.py, Paper II section 3.1).

The catalog is HYPERLEDA II, the homogenized HI data, table VII/238/hidat, served by the VizieR mirror at the
Harvard-Smithsonian Center for Astrophysics. It is queried here, and not the HyperLEDA site, whose robots file
disallows automated queries. For every SPARC galaxy a cone search of 180 arcsec is made at the position of its Sesame
reply (fetch_sesame.py), two seconds apart. It supplies the velocity of the galaxies that neither the resolver nor
ALFALFA gives (fetch_vizier.py), and a second source for the others. Each reply is saved as fetched under
vizier/<galaxy>__leda-hi.tsv in the data folder, and vizier-hi-manifest.tsv in the data folder lists every request
with its address, file, bytes, SHA-256 and time of fetch. The data folder is fetched by the reader with the four
fetch scripts, and its SHA256SUMS checks a new fetch against the replies of 2026-11-22 that Paper II used.

Run from the repository root after fetch_vizier.py (about 6 minutes, 175 requests; standard library only):
    python rotation-curves/fetch_vizier_hi.py data/sparc-velocities-2026-11-22
"""
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_vizier import AGENT, BASE, sesame_state  # noqa: E402

SRC, RAD, COLS = "VII/238/hidat", 180, "_r,PGC,AName,RAJ2000,DEJ2000,VHI,e_VHI,u_VHI"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    sub = os.path.join(dest, "vizier")
    if any(f.endswith("__leda-hi.tsv") for f in os.listdir(sub)):
        sys.exit("replies of this catalog are already in " + sub + "; a later fetch goes into a new dated folder")
    rows = ["galaxy\tcatalog\turl\tfile\tbytes\tsha256\tfetched_utc"]
    for gal, (pos, _) in sesame_state(dest).items():
        if pos is None:
            continue
        q = {"-source": SRC, "-c": f"{pos[0]:.6f} {pos[1]:+.6f}", "-c.rs": str(RAD), "-out": COLS, "-sort": "_r"}
        url = BASE + urllib.parse.urlencode(q, quote_via=urllib.parse.quote, safe="/,_")
        req = urllib.request.Request(url, headers={"User-Agent": AGENT})
        body = b""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read()
                break
            except Exception as e:
                print(gal, "attempt", attempt + 1, "failed:", e, flush=True)
                time.sleep(10)
        fn = f"{gal}__leda-hi.tsv"
        with open(os.path.join(sub, fn), "wb") as fh:
            fh.write(body)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows.append("\t".join([gal, SRC, url, "vizier/" + fn, str(len(body)), hashlib.sha256(body).hexdigest(), stamp]))
        nrow = sum(1 for ln in body.decode("utf-8", "replace").splitlines() if ln and not ln.startswith("#"))
        print(f"{gal:12s} {len(body):6d} bytes, {max(nrow - 3, 0)} rows", flush=True)
        time.sleep(2.0)
    with open(os.path.join(dest, "vizier-hi-manifest.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
