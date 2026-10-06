# -*- coding: utf-8 -*-
"""E7930 - K2 getrennt nach Restklassen p mod 6 (Kugelarithmetik), p-gleichmaessig ab P0(Klasse).
Zweiter Beweisweg neben der klassenfreien Paritaets-Minorante (E7927): garantierte Teiler d/m, m | gcd(d,6).
  rho >= 1 : (w_cl - p)/p = (h-1) - h*rho/p, steigt in p -> Wert bei p = P0 benutzt   (h = sum_{m|gcd(d,6)} 1/m)
  rho = -1 : (h-1) + h/p >= h-1                                                           (m = 1 zulaessig, i = p+1)
  rho <= -2: (h'-1) + h'|rho|/p >= h'-1  (h' ohne m = 1, weil d > p+1)
  -t/p >= -t/P0.   Klasse 5: P0 = 29, Klasse 1: P0 = 31 (d/6 >= 2 fuer alle rho <= 19).  Die Gewichtsungleichungen: E7928 (c).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2lib import *
from flint import arb, fmpq, ctx
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "E7930_klassen_mod6__20261006_2000.txt")
lines = []
def out(s=""):
    print(s); lines.append(s)

R = 19
def coeffs(c, P0):
    a = {}
    for rho in range(-R, R + 1):
        if rho == 0: continue
        dr = (c - rho) % 6
        g6 = gcd(dr, 6) if dr != 0 else 6
        ms = [m for m in (1, 2, 3, 6) if g6 % m == 0]
        if rho >= 1:
            h = sum(fmpq(1, m) for m in ms)
            a[rho] = (h - 1) - h * rho / fmpq(P0)
        elif rho == -1:
            h = sum(fmpq(1, m) for m in ms)
            a[rho] = h - 1
        else:
            h = sum(fmpq(1, m) for m in ms if m >= 2)
            a[rho] = h - 1
    return a

def Lcl(c, P0, t):
    a = coeffs(c, P0)
    tot = arb(0)
    for rho, co in a.items():
        tot += arb(co) * g(rho, t)
    return tot - TR(R, t) - t / P0

THR = arb("0.0015")
A, B = arb(fmpq(18, 100)), arb(1)
res = {}
for c, P0 in [(5, 29), (1, 31)]:
    r = certify_adaptive(R, None, A, B, THR, fun=lambda t, c=c, P0=P0: Lcl(c, P0, t))
    v18 = Lcl(c, P0, A)
    # Rastermin
    mn = min((float(Lcl(c, P0, arb(fmpq(180000 + k * 4100, 1000000))).mid()), k) for k in range(0, 201))
    out(f"Klasse p = {c} mod 6, p >= {P0}: Zertifikat Lcl > 0,0015 auf [0,18;1]: ok={r['ok']}, Kaesten {r['boxes']}, Lcl(0,18) = {v18.str(10)}, Rastermin {mn[0]:.6f} bei t = {0.18 + mn[1]*0.0041:.4f}")
    res[c] = (r['ok'], float(v18.mid()), mn[0])
# Gegenprobe gegen die Direktsummen aus E7929 (p >= 29)
ch = json.load(open(os.path.join(HERE, "out", "E7929_chain.json")))
viol = 0; cnt = 0; minsl = 9
cache = {}
for p, t, Pp, Lp in ch["rows"]:
    if p < 29: continue
    c = p % 6
    key = (c, t)
    if key not in cache:
        P0 = 29 if c == 5 else 31
        cache[key] = float(Lcl(c, P0, arb(fmpq(int(round(t * 1000000)), 1000000))).mid())
    L2 = cache[key]
    cnt += 1
    if Pp < L2 - 1e-12: viol += 1
    minsl = min(minsl, Pp - L2)
out(f"Klassen-Minorante gegen {cnt} Direktsummen-Zellen (p >= 29, E7929): {viol} Verletzungen, kleinster Schlupf {minsl:.5f}")
# Kontrolle: Klasse vertauscht muss verletzen (Mutant)
cache2 = {}
vv = 0
for p, t, Pp, Lp in ch["rows"]:
    if p < 29: continue
    c = 1 if p % 6 == 5 else 5
    P0 = 29 if c == 5 else 31
    if (c, t) not in cache2:
        cache2[(c, t)] = float(Lcl(c, P0, arb(fmpq(int(round(t * 1000000)), 1000000))).mid())
    if Pp < cache2[(c, t)] - 1e-12: vv += 1
out(f"Mutant 'Klassen vertauscht': {vv} Verletzungen")
ok = all(v[0] for v in res.values()) and viol == 0 and vv > 0
out(f"ERGEBNIS E7930: {'GRUEN' if ok else 'ROT'}")
open(LOG, "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump({str(k): v for k, v in res.items()}, open(os.path.join(HERE, "out", "E7930_result.json"), "w"))
sys.exit(0 if ok else 1)
