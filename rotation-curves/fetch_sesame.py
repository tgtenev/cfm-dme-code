"""Fetch the position and the heliocentric velocity of every SPARC galaxy from the Sesame name resolver, the first of
the four fetches behind the test of the direction of a new layer (layer_direction_test.py, Paper II section 3.1).

The resolver is the one of the VizieR mirror at the Harvard-Smithsonian Center for Astrophysics, which answers from
the SIMBAD database mirrored there and falls back to NED and VizieR. The CDS hosts in Strasbourg did not answer on
2026-11-22, and the robots files of NED's services and of HyperLEDA disallow automated queries, so neither is queried
directly.

Each galaxy is queried by one or two names (the SPARC name and the form the resolvers know), one request at a time,
two seconds apart. Every reply is saved as fetched, one file per request, under sesame/ in the data folder, and
sesame-manifest.tsv in the data folder lists every request: galaxy, try, query, address, file, bytes, SHA-256 and time
of fetch. Nothing in the replies is parsed here; layer_direction_test.py reads them.

The data folder is not redistributed; the reader fetches it with this script and the three that follow. The replies
that Paper II used were fetched on 2026-11-22 into data/sparc-velocities-2026-11-22/, whose SHA256SUMS gives the
checksum of every reply and manifest, so a new fetch into that folder can be checked with sha256sum -c SHA256SUMS run
inside it. The resolver's answers may change with time, and the checksums show where. The fetch refuses a data folder
whose sesame/ already holds replies.

Run from the repository root (about 7 minutes, 195 requests; standard library only), first of the four fetches:
    python rotation-curves/fetch_sesame.py data/sparc-velocities-2026-11-22 data/sparc/SPARC_Lelli2016c.mrt
"""
import hashlib
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxp/SNV?"
AGENT = "sparc-velocities-fetch/1.0 (python urllib)"


def sparc_names(path):
    names = []
    rule = 0
    for line in open(path, encoding="ascii", errors="replace"):
        s = line.rstrip("\r\n")
        if re.fullmatch(r"-+", s):
            rule += 1
            continue
        if rule < 4 or not s.strip():
            continue
        f = s.split()
        if len(f) >= 18 and re.fullmatch(r"\d+", f[17]):
            names.append(f[0])
    return names


def query_names(n):
    """The names to try, in order: the form the resolvers know first, the SPARC name second."""
    out = []
    m = re.fullmatch(r"(NGC|IC|UGC|DDO|PGC)0*(\d+)", n)
    if m:
        out.append(f"{m.group(1)} {int(m.group(2))}")
    m = re.fullmatch(r"UGCA0*(\d+)", n)
    if m:
        out.append(f"UGCA {int(m.group(1))}")
    m = re.fullmatch(r"ESO(\d{3})-G0*(\d+)", n)
    if m:
        out.append(f"ESO {m.group(1)}-{int(m.group(2))}")
    m = re.fullmatch(r"([DF])(\d{3})-(V?)(\d+)", n)
    if m:
        out.append(f"LSBC {m.group(1)}{m.group(2)}-{m.group(3)}{int(m.group(4)):02d}")
    if n == "KK98-251":
        out.append("[KK98] 251")
    if n == "CamB":
        out.append("Cam B")
    if n not in out:
        out.append(n)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dest, table = sys.argv[1], sys.argv[2]
    sub = os.path.join(dest, "sesame")
    os.makedirs(sub, exist_ok=True)
    if os.listdir(sub):
        sys.exit("the folder " + sub + " is not empty; a later fetch goes into a new dated folder")
    rows = ["galaxy\ttry\tquery\turl\tfile\tbytes\tsha256\tfetched_utc"]
    for n in sparc_names(table):
        for k, q in enumerate(query_names(n), 1):
            url = BASE + urllib.parse.quote(q)
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            body = b""
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=90) as r:
                        body = r.read()
                    break
                except Exception as e:                      # retried twice, then recorded as empty
                    print(n, q, "attempt", attempt + 1, "failed:", e, flush=True)
                    time.sleep(10)
            fn = f"{n}__{k}.xml"
            with open(os.path.join(sub, fn), "wb") as fh:
                fh.write(body)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rows.append("\t".join([n, str(k), q, url, "sesame/" + fn, str(len(body)),
                                   hashlib.sha256(body).hexdigest(), stamp]))
            found = b"<jradeg>" in body
            vel = (b"<Vel>" in body) or (b"<z>" in body)
            print(f"{n:12s} try {k} {q!r:22s} {len(body):6d} bytes  position {'yes' if found else 'no '}  velocity "
                  f"{'yes' if vel else 'no'}", flush=True)
            time.sleep(2.0)
            if found and vel:
                break
    with open(os.path.join(dest, "sesame-manifest.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
