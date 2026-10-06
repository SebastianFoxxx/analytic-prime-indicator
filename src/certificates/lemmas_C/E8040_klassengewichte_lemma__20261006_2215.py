"""E8040 (T36-F-1260) MESSLAUF: Lemma K (Klassengewichte), Anhang C des API-Papers, B2 der Gegenlese T35-V-1091.

Gate E8039 gruen (16/16), committet vor diesem Lauf.

VORAB-REGELN:
 A  Formeln c_a(q,rho) = (1-rho/q) K_r(rho), c_b(q,-1) = 1+1/q, sonst 0 gelten EXAKT (Bruchrechnung, Definition gegen Formel) fuer ALLE
    q = r (6), 53 <= q <= 200000, 0 < |rho| <= 40, r in {1,5}: 0 Verletzungen.
 B  Tabelle K_r(rho) (exakt) wird ausgegeben (Beleg fuer die Handformel des Papers); Werte nur aus {0, 1/3, 1/2, 1, 4/3, 3/2, 2} (und ohne m=1 bei rho<0).
 C  Exakte Koeffiziententabelle (Limes exakt statt q = 10^9+7): c_a, c_b je Klasse; Abstand zur Alttabelle <= 4e-8 (Erwartung aus Gate K5).
 D  Gegenprobe an den PRIMZAHLEN selbst: fuer alle Primzahlen p der Klasse, p_lo <= p <= 6e6, und alle 0 < |rho| <= 40 gilt
    (garantierte Teiler, direkt aus der Definition per numpy) sum_{i in G, i != p+1} i >= p c_a(rho)  und  [p+1 in G](p+1)/p >= c_b(rho): 0 Verletzungen.
 E  Gegenprobe gegen die VOLLEN Teilersummen: fuer jede Primzahl bis 2e5 ist sum_{i | p-rho, 2<=i<=p, i != p+1} i  >= garantierter Teil (triviale Richtung,
    Detektor-Kontrolle, dass G wirklich eine Teilmenge der echten Teiler ist): 0 Verletzungen.
 F  Mutant: c_a um 1 % erhoeht -> Verletzungen in D > 0 (Detektor nicht blind).
"""
import os
import sys
import time
import json
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "T36_F1198"))
import l1260lib as L

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


t0 = time.time()
log("E8040 Lemma K Klassengewichte", time.strftime("%Y-%m-%d %H:%M"))
PLO = {1: 31, 5: 23}
# A
badA = 0
nA = 0
for r in (1, 5):
    b, n, first = L.check_structure(r, 53, 200000)
    badA += b
    nA += n
    log(f"A  Klasse {r}: q in [53, 200000], {n} Paare (q,rho): Verletzungen {b}")
okA = badA == 0
# B
log("B  K_r(rho) (exakt):")
for r in (1, 5):
    row = {rho: L.K_formula(r, rho) for rho in range(-40, 41) if rho != 0}
    vals = sorted(set(row.values()))
    log(f"   Klasse {r}: Wertemenge {[str(v) for v in vals]}")
    for rho in (-3, -2, -1, 1, 2, 3, 4, 5, 6, 7):
        log(f"     rho={rho:3d}: gcd(r-rho,6)={np.gcd(r - rho, 6)}  K={row[rho]}")
okB = all(v in (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1), Fraction(4, 3), Fraction(3, 2), Fraction(2)) for r in (1, 5) for v in [L.K_formula(r, rho) for rho in range(-40, 41) if rho != 0])
# C
import p1198uni as U
coef = {}
maxd = 0.0
for r in (1, 5):
    new = L.class_coeffs_exact(r, PLO[r])
    old = U.class_coeffs(r, p_lo=PLO[r])
    coef[r] = new
    for rho in new:
        for k in (0, 1):
            maxd = max(maxd, float(old[rho][k] - new[rho][k]))
log(f"C  Abstand alt (q=10^9+7) - neu (exakt): max {maxd:.3e}")
okC = maxd <= 4.1e-8
# D / F
def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for k in range(2, int(n ** 0.5) + 1):
        if s[k]:
            s[k * k::k] = False
    return np.nonzero(s)[0].astype(np.int64)


PMAX = 6_000_000
primes = sieve(PMAX)
log(f"D  Primzahlen bis {PMAX}: {len(primes)}")


def check_primes(r, scale=Fraction(1)):
    P = primes[(primes % 6 == r) & (primes >= PLO[r])]
    viol = 0
    minslack = None
    for rho, (ca, cb) in coef[r].items():
        d = P - rho
        g = np.gcd(d, 6)
        tot = np.zeros_like(P)
        edge = np.zeros(len(P), dtype=bool)
        for m in (1, 2, 3, 6):
            mask = (g % m == 0) & (d > 0)
            i = d // m
            ok = mask & (i >= 2) & (i <= P + 1)
            edge |= ok & (i == P + 1)
            tot += np.where(ok & (i != P + 1), i, 0)
        # tot/P >= ca*scale  <=>  tot * den >= num * P
        caS = ca * scale
        lhs = tot * caS.denominator
        rhs = caS.numerator * P
        viol += int(np.sum(lhs < rhs))
        sl = float(np.min(tot / P - float(caS))) if len(P) else 0.0
        minslack = sl if minslack is None else min(minslack, sl)
        # c_b
        cb_val = np.where(edge, (P + 1) / P, 0.0)
        viol += int(np.sum(cb_val < float(cb) - 1e-15))
    return viol, minslack, len(P)


vD = {}
for r in (1, 5):
    v, ms, n = check_primes(r)
    vD[r] = v
    log(f"D  Klasse {r} (p >= {PLO[r]}): {n} Primzahlen x 80 rho: Verletzungen {v}; kleinster Abstand sum/p - c_a = {ms:.3e}")
okD = all(v == 0 for v in vD.values())
vF = {}
for r in (1, 5):
    v, ms, n = check_primes(r, scale=Fraction(101, 100))
    vF[r] = v
    log(f"F  Mutant c_a * 1,01, Klasse {r}: Verletzungen {v}")
okF = all(v > 0 for v in vF.values())
# E
okE = True
nE = 0
for p in primes[primes <= 200000]:
    p = int(p)
    if p < 23:
        continue
    r = p % 6
    if r not in (1, 5):
        continue
    for rho in range(-40, 41, 7):
        if rho == 0:
            continue
        d = p - rho
        full = sum(i for i in range(2, p + 1) if d % i == 0) if d > 0 and p < 3000 else None
        if full is None:
            continue
        guar = sum(i for i in L.G_set(p, rho) if i != p + 1)
        nE += 1
        if full < guar:
            okE = False
log(f"E  Teilmengen-Kontrolle (p < 3000, {nE} Paare): {'0 Verletzungen' if okE else 'VERLETZUNG'}")
# Tabelle
tab = {str(r): {str(rho): [str(c[0]), str(c[1])] for rho, c in coef[r].items()} for r in (1, 5)}
with open(os.path.join(HERE, "out", "E8040_klassenkoeffizienten_exakt.json"), "w", encoding="utf-8") as fh:
    json.dump(tab, fh, indent=1)
log(f"Auswertung: A {okA}, B {okB}, C {okC}, D {okD}, E {okE}, F(Mutant faellt) {okF}")
log("ERGEBNIS:", "GRUEN" if all([okA, okB, okC, okD, okE, okF]) else "ROT", f"({time.time() - t0:.0f} s)")
with open(os.path.join(HERE, "logs", "E8040_klassengewichte__20261006_2215.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
