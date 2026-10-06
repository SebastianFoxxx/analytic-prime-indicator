# -*- coding: utf-8 -*-
"""E7928 - Lemma 2 (garantierte Teiler) EXAKT ganzzahlig, eigener Sieb (kein Import aus T36_F993/T35_V945).
Gewicht w_p(rho) = sum_{i | (p-rho), 2<=i<=p+1} i  (rho=p: alle i = 2..p+1).
Pruefungen fuer alle Primzahlen 23 <= p <= 2e6 und rho in [-19,19]:
  (a) Paritaets-Tabelle (Satz K2):   2(w-p) >= p*c2(rho) - 3*rho*[rho>=1]   mit c2 = 2*c_inf in {1,0,-1,-2}   (exakt, ganzzahlig)
  (b) w_p(0) = p
  (c) Klassen mod 6 (p >= 29): w >= w_cl = sum_{m | gcd(d,6), zulaessig} d/m  (d = p-rho; rho<=-2: m>=2; i=d/m>=2)
  Negativkontrollen: erfundenes Gewicht d + d/2 + d/3 ohne 3|d  und  M2 (c=+1/2 bei allen ungeraden rho).
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "E7928_lemma2_exakt__20261006_1930.txt")
lines = []
def out(s=""):
    print(s); lines.append(s)

N = 2_000_050
t0 = time.time()
sig = np.zeros(N + 1, dtype=np.int64)
for i in range(1, N + 1):
    sig[i::i] += i
out(f"sigma-Sieb bis {N}: {time.time()-t0:.1f}s")
is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
for i in range(2, int(N ** 0.5) + 1):
    if is_p[i]: is_p[i * i::i] = False
primes = np.nonzero(is_p)[0]
primes = primes[(primes >= 23) & (primes <= 2_000_000)]
out(f"Primzahlen 23..2e6: {len(primes)}")

def c2(rho):
    if rho == -1: return 1
    if rho >= 1: return 1 if rho % 2 else 0
    return -1 if rho % 2 else -2

viol_a = viol_b = viol_c = 0
ctrl_d3 = ctrl_m2 = 0
cells = 0
gcd6 = {r: np.gcd(r, 6) for r in range(6)}
for rho in range(-19, 20):
    d = primes - rho
    if rho == 0:
        w = sig[primes] - 1
        viol_b += int(np.count_nonzero(w != primes))
        continue
    # w = sigma(d) - 1 - [d > p+1]*d     (d>0 hier immer)
    w = sig[d] - 1 - np.where(d > primes + 1, d, 0)
    # (a)
    lhs = 2 * (w - primes)
    rhs = primes * c2(rho) - (3 * rho if rho >= 1 else 0)
    viol_a += int(np.count_nonzero(lhs < rhs))
    cells += len(primes)
    # Negativkontrollen (a)
    # erfundenes Gewicht d + d/2 + d/3 fuer rho>=1 (ohne 3|d zu pruefen) -> muss oft verletzt werden
    if rho >= 1:
        fake = d + d // 2 + d // 3
        ctrl_d3 += int(np.count_nonzero(w < fake))
    # M2: c = +1/2 bei allen ungeraden rho (inkl. rho <= -3): 2(w-p) >= p
    if rho % 2 != 0 and rho <= -3:
        ctrl_m2 += int(np.count_nonzero(2 * (w - primes) < primes))
    # (c) Klassen mod 6, p >= 29
    msk = primes >= 29
    pp = primes[msk]; dd = d[msk]; ww = w[msk]
    g = np.gcd(dd, 6)
    wcl = np.zeros_like(dd)
    for m in (1, 2, 3, 6):
        ok = (g % m == 0)
        if rho <= -2 and m == 1: ok = ok & False
        ok = ok & (dd // m >= 2) & (dd // m <= pp + 1)
        wcl = wcl + np.where(ok, dd // m, 0)
    viol_c += int(np.count_nonzero(ww < wcl))

out(f"(a) Paritaets-Tabelle: {cells} Zellen, {viol_a} Verletzungen")
out(f"(b) w_p(0) = p: {viol_b} Verletzungen ueber {len(primes)} Primzahlen")
out(f"(c) Klassen-mod-6-Gewichte (p>=29): {viol_c} Verletzungen")
out(f"Negativkontrollen: erfundenes d/3: {ctrl_d3} Verletzungen; M2 (c=+1/2 bei ungeraden rho<=-3): {ctrl_m2} Verletzungen")
ok = viol_a == 0 and viol_b == 0 and viol_c == 0 and ctrl_d3 > 0 and ctrl_m2 > 0
out(f"ERGEBNIS E7928: {'GRUEN' if ok else 'ROT'}  ({time.time()-t0:.1f}s)")
open(LOG, "w", encoding="utf-8").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
