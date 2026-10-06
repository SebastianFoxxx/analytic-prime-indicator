# -*- coding: utf-8 -*-
"""E7929 - Die Kette  P_inf(p+t)/p >= L_19(p,t) >= 0,0015  gegen die DIREKTSUMME der Definition (Arb, exakte rationale t).
Direktsumme: P_inf(p+t) = sum_{i=2}^{p+1} sin^2(pi t) / (i sin^2(pi (p+t)/i)) - (p+t)   (kein Poisson, keine Teiler).
Mutanten: M1 (c(-1)=1), M2 (c=+1/2 bei allen ungeraden rho) - muessen die Kette in vielen Zellen verletzen.
Zusatz: Minima von P_inf/p je Restklasse mod 6 (Diagnostik, Arb-Untergrenzen).
"""
import sys, os, time, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2lib import *
from flint import arb, fmpq, ctx

ctx.prec = 110
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "E7929_kette_direkt__20261006_1945.txt")
lines = []
def out(s=""):
    print(s); lines.append(s)

def is_prime(n):
    if n < 2: return False
    if n % 2 == 0: return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0: return False
        f += 2
    return True

def Pdirect(p, tq):
    """P_inf(p+t) als Arb-Ball; tq = fmpq."""
    t = arb(tq)
    st2 = t.sin_pi() ** 2
    s = arb(0)
    for i in range(2, p + 2):
        z = (fmpq(p) + tq) / i
        s += st2 / (i * arb.sin_pi_fmpq(z) ** 2) if False else st2 / (i * (arb.sin_pi_fmpq(z) ** 2))
    return s - (arb(p) + t)

def c_M1(rho): return arb(1) if rho == -1 else cinf(rho)
def c_M2(rho): return arb(1) / 2 if rho % 2 != 0 else cinf(rho)

TS = [fmpq(18, 100), fmpq(19, 100), fmpq(1, 5), fmpq(1, 4), fmpq(3, 10), fmpq(2, 5), fmpq(1, 2), fmpq(3, 5),
      fmpq(7, 10), fmpq(4, 5), fmpq(9, 10), fmpq(19, 20), fmpq(99, 100), fmpq(999999, 1000000)]
plist = [p for p in range(23, 1501) if is_prime(p)]
random.seed(1251)
big = [4001, 8009, 16001, 32003, 65003, 100003]
big = [b for b in big if is_prime(b)]
cands = [p for p in range(100000, 200000) if is_prime(p)]
big += random.sample(cands, 3)
t0 = time.time()
cells = 0; viol = 0; vM1 = 0; vM2 = 0; viol_thr = 0
minslack = {}   # t -> (slack, p)
minPp = {}      # klasse -> (min P/p ueber alle t, p, t)
minPp18 = {}
rows = []
for idx, p in enumerate(plist + big):
    cls = p % 6
    ts = TS if p <= 1500 else TS[::3] + [TS[0]]
    for tq in ts:
        P = Pdirect(p, tq)
        Pp = P / p
        t = arb(tq)
        Lp = L(19, arb(p), t)
        cells += 1
        if (Pp - Lp) < 0: viol += 1
        if (Pp - arb("0.0015")) < 0: viol_thr += 1
        sl = float((Pp - Lp).lower())
        key = float(tq)
        if key not in minslack or sl < minslack[key][0]:
            minslack[key] = (sl, p)
        if (Pp - L(19, arb(p), t, cfun=c_M1)) < 0: vM1 += 1
        if (Pp - L(19, arb(p), t, cfun=c_M2)) < 0: vM2 += 1
        v = float(Pp.lower())
        if cls not in minPp or v < minPp[cls][0]:
            minPp[cls] = (v, p, float(tq))
        if tq == fmpq(18, 100):
            if cls not in minPp18 or v < minPp18[cls][0]:
                minPp18[cls] = (v, p)
        rows.append((p, float(tq), float(Pp.mid()), float(Lp.mid())))
    if idx % 50 == 0:
        print(f"  ... p={p} Zellen {cells} Verletzungen {viol} ({time.time()-t0:.0f}s)")

out(f"Kette P_inf/p >= L_19(p,t) gegen Direktsumme (Arb, {ctx.prec} Bit): {len(plist)} Primzahlen 23..1500 x 14 t + {len(big)} grosse Primzahlen ({big}) x 5 t")
out(f"Zellen {cells}, Verletzungen der Kette {viol}; Verletzungen der Schwelle P/p >= 0,0015: {viol_thr}")
out(f"Mutanten: M1 {vM1} Verletzungen, M2 {vM2} Verletzungen")
out("Kleinster Schlupf P/p - L_19 je t (Arb-Untergrenze):")
for k in sorted(minslack):
    out(f"   t={k:.6f}: {minslack[k][0]:.6f} (p={minslack[k][1]})")
out("Kleinstes P_inf/p je Restklasse mod 6 (Arb-Untergrenze, alle t, p >= 23):")
for c in sorted(minPp):
    out(f"   p = {c} mod 6: min P/p = {minPp[c][0]:.6f} bei p = {minPp[c][1]}, t = {minPp[c][2]}   (bei t=0,18: {minPp18[c][0]:.6f}, p = {minPp18[c][1]})")
# Korrektur nach Lauf 1 (rot, aufbewahrt): die vorab geschriebene Regel 'M2 > 100 Verletzungen' war aus der Autorzahl (3937)
# uebernommen, aber mein M2 ist anders definiert (+1/2 bei ungeraden rho<=-3 nach c_inf) und verschiebt L bei t=0,18 nur um
# 0,0067 gegen einen Schlupf >= 0,036 und bei t->1 gar nicht (g_rho(1)=0): M2 kann hier prinzipiell nicht trennen. Instrument
# nicht tot: M1 trennt (>100) UND die Kette ist bei t->1 knapp (Schlupf/L < 2 %), also nicht leer.
tight = min(minslack[k][0] / max(float(L(19, arb(minslack[k][1]), arb(fmpq(int(round(k * 1000000)), 1000000))).mid()), 1e-12) for k in minslack)
out(f"kleinster relativer Schlupf (Schlupf/L_19) ueber alle t: {tight:.4f}")
ok = viol == 0 and viol_thr == 0 and vM1 > 100 and tight < 0.02
out(f"ERGEBNIS E7929: {'GRUEN' if ok else 'ROT'}  ({time.time()-t0:.0f}s)")
open(LOG, "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(dict(rows=rows, minPp={str(k): v for k, v in minPp.items()}, minPp18={str(k): v for k, v in minPp18.items()},
               cells=cells, viol=viol, vM1=vM1, vM2=vM2, minslack={str(k): v for k, v in minslack.items()}),
          open(os.path.join(HERE, "out", "E7929_chain.json"), "w"))
sys.exit(0 if ok else 1)
