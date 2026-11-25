"""Reader of the globular-cluster database's rvfit figures (vector PDF) and of its velocity-dispersion table.

A module, with no output of its own, imported by globular-clusters/gc_dynamical.py. It reads
data/baumgardt-gc/pages/rvfit/<cluster>_rvfit.pdf and data/baumgardt-gc/pages/veldis.html, which the reader
fetches from the database (they are not redistributed).

model_curve(stem, vbins) returns (log10 R_arcsec array, sigma array) for the red N-body curve.
Calibration:
  y: sigma = top * (Y - Y0) / (Ytop - Y0); 'top' is the round axis value implied by the plotted
     points matched to the veldis bins (each implied top must lie within 1.5% of a round value).
  x: the long ticks (length 125 units) on the zero line are decades; the decade of the first long
     tick is chosen so that the plotted points fall on the veldis radii.
"""
import math
import os
import re
import html as htmlmod
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "..", "data", "baumgardt-gc", "pages")
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")
TOK = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)|[A-Za-z*'\"]+|/[^\s/\[\]()<>]+|\[|\]")
ROUND = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def content_stream(path):
    raw = open(path, "rb").read()
    best = ""
    for m in re.finditer(rb"stream\r?\n", raw):
        st = m.end()
        en = raw.find(b"endstream", st)
        try:
            txt = zlib.decompress(raw[st:en]).decode("latin-1")
        except Exception:
            try:
                txt = zlib.decompressobj().decompress(raw[st:en]).decode("latin-1")
            except Exception:
                continue
        if (" l\n" in txt or " l " in txt) and len(txt) > len(best):
            best = txt
    return best


def parse(path):
    c = content_stream(path)
    stack, paths, cur = [], [], []
    stroke = fill = (0.0, 0.0, 0.0)
    for t in TOK.findall(c):
        if NUM.fullmatch(t):
            stack.append(float(t))
            continue
        op = t
        if op.startswith("/") or op in ("[", "]"):
            stack = []
            continue
        if op == "m":
            cur.append([(stack[-2], stack[-1])])
        elif op in ("l", "c") and cur:
            cur[-1].append((stack[-2], stack[-1]))
        elif op == "re":
            x, y, w, h = stack[-4:]
            cur.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
        elif op == "RG":
            stroke = tuple(stack[-3:])
        elif op == "rg":
            fill = tuple(stack[-3:])
        elif op in ("S", "s"):
            paths.append(("S", stroke, cur))
            cur = []
        elif op in ("f", "f*", "F", "B", "b", "B*", "b*"):
            paths.append(("F", fill, cur))
            cur = []
        elif op == "n":
            cur = []
        stack = []
    return paths


def geometry(stem):
    P = parse(os.path.join(INP, "rvfit", stem + "_rvfit.pdf"))
    red = [sp for k, col, sps in P if k == "S" and col == (1.0, 0.0, 0.0) for sp in sps if len(sp) > 5]
    curve = max(red, key=len)
    cx = np.array([p[0] for p in curve])
    cy = np.array([p[1] for p in curve])
    o = np.argsort(cx)
    cx, cy = cx[o], cy[o]
    hl = []
    for k, col, sps in P:
        if k == "S" and col == (0.0, 0.0, 0.0):
            for sp in sps:
                for (x1, y1), (x2, y2) in zip(sp[:-1], sp[1:]):
                    if abs(y1 - y2) < 0.01 and abs(x2 - x1) > 3000:
                        hl.append(round(y1, 1))
    ys = sorted(set(hl))
    Y0 = max(y for y in ys if y <= cy.min() + 0.5)
    Ytop = min(y for y in ys if y > cy.max())
    circ = []
    for k, col, sps in P:
        if k == "F" and col == (0.0, 0.0, 0.0):
            for sp in sps:
                xs = [p[0] for p in sp]
                yy = [p[1] for p in sp]
                if len(sp) >= 8 and 20 < max(xs) - min(xs) < 150 and 20 < max(yy) - min(yy) < 150:
                    X, Y = 0.5 * (max(xs) + min(xs)), 0.5 * (max(yy) + min(yy))
                    if Y0 + 1 < Y < Ytop:
                        circ.append((X, Y))
    longt = []
    for k, col, sps in P:
        if k == "S":
            for sp in sps:
                for (x1, y1), (x2, y2) in zip(sp[:-1], sp[1:]):
                    if abs(x1 - x2) < 0.01 and abs(min(y1, y2) - Y0) < 0.6 and 100 < abs(y1 - y2) < 150:
                        longt.append(round(x1, 1))
    longt = sorted(set(longt))
    return dict(cx=cx, cy=cy, Y0=Y0, Ytop=Ytop, circles=sorted(circ), longticks=longt)


def calibrate(stem, vbins):
    g = geometry(stem)
    lt = g["longticks"]
    dec = float(np.median(np.diff(lt)))
    rv = [b for b in vbins if b["type"] == "RV"]
    # y: implied round top from the circles matched to bins (sigma ordering)
    tops = []
    for X, Y in g["circles"]:
        for b in rv:
            t = b["s"] * (g["Ytop"] - g["Y0"]) / (Y - g["Y0"])
            r = min(ROUND, key=lambda v: abs(math.log(t / v)))
            if abs(math.log(t / r)) < 0.015:
                tops.append(r)
    top = max(set(tops), key=tops.count) if tops else None
    # x: decade of the first long tick from the circle positions
    best = None
    for e in range(-1, 3):
        first = 10.0 ** e
        mism = 0.0
        for X, Y in g["circles"]:
            lr = math.log10(first) + (X - lt[0]) / dec
            if rv:
                mism += min(abs(lr - math.log10(b["r"])) for b in rv)
        if best is None or mism < best[0]:
            best = (mism, first)
    first = best[1]
    return g, top, first, dec


def model_curve(stem, vbins, top_override=None, first_override=None):
    g, top, first, dec = calibrate(stem, vbins)
    if top_override is not None:
        top = top_override
    if first_override is not None:
        first = first_override
    lr = math.log10(first) + (g["cx"] - g["longticks"][0]) / dec
    sig = top * (g["cy"] - g["Y0"]) / (g["Ytop"] - g["Y0"])
    return lr, sig, top, first


def parse_veldis(path=None):
    path = path or os.path.join(INP, "veldis.html")
    t = open(path, encoding="utf-8", errors="replace").read()
    res, cur = {}, None
    for tr in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", t):
        cells = [htmlmod.unescape(re.sub(r"(?s)<[^>]+>", " ", x)).strip() for x in re.findall(r"(?is)<td[^>]*>(.*?)</td>", tr)]
        if len(cells) < 7:
            continue
        nm = re.sub(r"\s+", " ", cells[0]).strip()
        if nm:
            cur = nm
        try:
            b = dict(r=float(cells[1]), n=int(cells[2]) if cells[2].strip() else None, s=float(cells[3]),
                     up=float(cells[4]), lo=float(cells[5]), type=cells[6].strip())
        except ValueError:
            continue
        res.setdefault(cur, []).append(b)
    return res
