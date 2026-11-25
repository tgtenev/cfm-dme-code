"""Fetch, from VizieR, the heliocentric velocities that the Sesame replies lack, and a second source for the rest: the
second of the four fetches behind the test of the direction of a new layer (layer_direction_test.py, Paper II
section 3.1).

For every SPARC galaxy, at the position of its Sesame reply (fetch_sesame.py), two cone searches are made on the
VizieR mirror at the Harvard-Smithsonian Center for Astrophysics:
  - the Updated Nearby Galaxy Catalog, Karachentsev, Makarov & Kaisina (2013), J/AJ/145/101/catalog, within
    180 arcsec, for every galaxy: heliocentric velocity, distance and its method, as a check of the resolver;
  - the ALFALFA extragalactic HI source catalog, Haynes et al. (2018), J/ApJ/861/49/table2, within 90 arcsec, for
    the galaxies whose Sesame reply holds no velocity.
Each reply is saved as fetched under vizier/ in the data folder, two seconds apart, and vizier-manifest.tsv in the
data folder lists every request: galaxy, catalog, address, file, bytes, SHA-256 and time of fetch. Nothing is parsed
here beyond the position in the Sesame reply. The data folder is fetched by the reader with the four fetch scripts,
and its SHA256SUMS checks a new fetch against the replies of 2026-11-22 that Paper II used.

Run from the repository root after fetch_sesame.py (about 7 minutes, 195 requests; standard library only):
    python rotation-curves/fetch_vizier.py data/sparc-velocities-2026-11-22
"""
import hashlib
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
AGENT = "sparc-velocities-fetch/1.0 (python urllib)"
CATS = {
    "ungc": ("J/AJ/145/101/catalog", 180, "_r,Name,RAJ2000,DEJ2000,HRV,Dist,f_Dist,Vlg"),
    "alfalfa": ("J/ApJ/861/49/table2", 90, "_r,AGC,Name,RAJ2000,DEJ2000,RAO,DEO,Vhel,W50,e_W50,HI"),
}


def sesame_state(dest):
    """Per galaxy, from the manifest and the replies: (ra, dec) of the last reply with a position, and whether any
    reply holds a velocity."""
    st = {}
    for line in open(os.path.join(dest, "sesame-manifest.tsv"), encoding="utf-8").read().splitlines()[1:]:
        f = line.split("\t")
        gal, path = f[0], f[4]
        txt = open(os.path.join(dest, path), encoding="utf-8", errors="replace").read()
        ra = re.search(r"<jradeg>\s*([-+0-9.]+)\s*</jradeg>", txt)
        de = re.search(r"<jdedeg>\s*([-+0-9.]+)\s*</jdedeg>", txt)
        vel = ("<Vel>" in txt) or ("<z>" in txt)
        pos, had = st.get(gal, (None, False))
        if ra and de:
            pos = (float(ra.group(1)), float(de.group(1)))
        st[gal] = (pos, had or vel)
    return st


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dest = sys.argv[1]
    sub = os.path.join(dest, "vizier")
    os.makedirs(sub, exist_ok=True)
    if os.listdir(sub):
        sys.exit("the folder " + sub + " is not empty; a later fetch goes into a new dated folder")
    rows = ["galaxy\tcatalog\turl\tfile\tbytes\tsha256\tfetched_utc"]
    for gal, (pos, had) in sesame_state(dest).items():
        if pos is None:
            print(gal, "no position; skipped", flush=True)
            continue
        for key in (("ungc",) if had else ("ungc", "alfalfa")):
            src, rad, cols = CATS[key]
            q = {"-source": src, "-c": f"{pos[0]:.6f} {pos[1]:+.6f}", "-c.rs": str(rad), "-out": cols, "-sort": "_r"}
            url = BASE + urllib.parse.urlencode(q, quote_via=urllib.parse.quote, safe="/,_")
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            body = b""
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        body = r.read()
                    break
                except Exception as e:
                    print(gal, key, "attempt", attempt + 1, "failed:", e, flush=True)
                    time.sleep(10)
            fn = f"{gal}__{key}.tsv"
            with open(os.path.join(sub, fn), "wb") as fh:
                fh.write(body)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rows.append("\t".join([gal, src, url, "vizier/" + fn, str(len(body)), hashlib.sha256(body).hexdigest(), stamp]))
            nrow = sum(1 for ln in body.decode("utf-8", "replace").splitlines() if ln and not ln.startswith("#"))
            print(f"{gal:12s} {key:8s} {len(body):6d} bytes, {max(nrow - 3, 0)} rows", flush=True)
            time.sleep(2.0)
    with open(os.path.join(dest, "vizier-manifest.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
