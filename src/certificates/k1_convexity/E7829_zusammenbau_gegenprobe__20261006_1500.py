"""E7829 (T36-F-1198, R2) ZUSAMMENBAU-GEGENPROBE am vollen P_sigma (mpmath, direkt aus der Definition, 30 Stellen; KEIN Import aus p1198lib).

Satz-Kandidat (K3'): p >= 5 prim, kappa >= 3,25 (p+1) ln p  =>  P_sigma(.;kappa) hat in (p, p+1) genau eine Nullstelle z, z < 0,18, P_sigma'(z) > 0.
Beweisschritte (Papier): (S1) f(0) = P_sigma(p) < 0 (geschlossene Form); (S2) f'' > 0 auf (0, 0,18] (K1'); (S3) f(0,18) > 0 und f > 0 auf [0,18; 1) (K2');
 (S4) Konvexitaet + f(0)<0<f(0,18) => genau ein Vorzeichenwechsel in (0, 0,18), f'(z) > (f(z)-f(0))/z > 0.
VORAB-REGELN (festgeschrieben vor dem Lauf):
 R1 Je Zelle (p, m), kappa = m * 3,25 (p+1) ln p: S1 f(0) = -p/(1+exp(2 kappa/(p+1))) stimmt mit der Integer-Formel (Paper Prop. 8.2, Teilersumme) auf 1e-25 relativ ueberein, ist < 0 und f(p+1e-10) liegt innerhalb 1,0001e-10 davon (Lauf 3: t = 1e-15 war bei 30 Stellen Rauschen, rot) (Stetigkeit bei t -> 0+).
    [R1 wurde nach Lauf 1 als INSTRUMENTFEHLER korrigiert (C8): die erste Fassung wertete bei t = 1e-22 aus und zog t ab; bei f(0) ~ 1e-45..1e-96 und 30 Stellen ist das Ausloeschung. Lauf 1 aufbewahrt: logs/E7829_..._lauf1_rot.txt; die Regeln R2-R6 waren davon nicht betroffen.]
 R2 S2 numerisch: f''(p+t) > 0 an allen t des Gitters (t = 1e-6, 1e-4, 1e-3, dann 120 gleichmaessig bis 0,18); Erwartung: 0 Verletzungen.
 R3 S3: f(p+0,18) > 0 und f > 0 auf einem Gitter von 400 Punkten in [0,18; 1) sowie bei t = 1 - 10^-k, k = 3..8: Erwartung 0 Verletzungen (p = 5 knapp: min P/p ~ 2,5e-4 laut T36-F-1195 E7793).
 R4 S4: genau ein Vorzeichenwechsel auf dem Gesamtgitter (0,1), die Nullstelle z per Bisektion (30 Stellen), z < 0,18, f'(z) > 0 (mp.diff) und f'(z) > -f(0)/z... (die Konvexitaets-Abschaetzung, scharf geprueft).
 R5 Zellen: p in {5,7,11,13,17,19,23,29,31,37,101,499,997}, m in {1; 1,7; 3; 10}; 52 Zellen.  Erwartung: alle gruen.
 R6 Mutanten: (a) kappa = 0,3 * kappa_min bei p = 5 (unterhalb des K2'-Bereichs: f(0,18) < 0 oder mehr als eine Nullstelle erwartet); (b) p = 9 (zusammengesetzt, kappa wie fuer p = 9): S2 oder die Nullstellenzahl bricht.  Erwartung: mindestens eine der Regeln R2-R4 faellt je Mutante.
"""
import math
import os
import sys
import time

import mpmath as mp

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def phi(i, x, k):
    return 1 / (1 + mp.exp(2 * k * (mp.mpf(i) / (x + 1) - 1)))


def Pf(x, p, k):
    x = mp.mpf(x)
    I = int(p + 3 + 50 * (p + 1.3) / (2 * k)) + 3
    s = mp.mpf(0)
    for i in range(2, I + 1):
        s += phi(i, x, k) * i * (mp.sin(mp.pi * x) / (i * mp.sin(mp.pi * x / i))) ** 2
    return s - x


def cell(p, k, tag=""):
    k = mp.mpf(k)
    f0_closed = -p / (1 + mp.exp(2 * k / (p + 1)))
    # R1' (Instrumentkorrektur, siehe Log lauf1_rot): Integer-Formel des Papers (Teilersumme) statt Auswertung bei t = 1e-22 (Ausloeschung)
    with mp.workdps(320):  # f(0) ~ e^{-449} bei p=997, m=10: Hochpraezision gegen Ausloeschung (Lauf 2 war bei 30 Stellen rot)
        kk = mp.mpf(k)
        f0_int = sum(i * phi(i, mp.mpf(p), kk) for i in range(2, p + 1) if p % i == 0) - p
        f0_cl = -p / (1 + mp.exp(2 * kk / (p + 1)))
        r1 = abs(f0_int - f0_cl) / abs(f0_cl)
    cont = abs(Pf(mp.mpf(p) + mp.mpf('1e-10'), p, k) - f0_closed)  # Stetigkeit bei t -> 0+: |f(1e-10)-f(0)| <= 1,0001e-10 (|f'(0)| < 1; Lauf 3 mit t=1e-15 war bei 30 Stellen rot: Rauschen 1e-30/1e-15)
    ok1 = (r1 < mp.mpf('1e-25')) and f0_closed < 0 and cont < mp.mpf('1.0001e-10')
    ts = [mp.mpf('1e-6'), mp.mpf('1e-4'), mp.mpf('1e-3')] + [mp.mpf(0.18) * j / 120 for j in range(1, 121)]
    viol2 = 0
    for t in ts:
        d2 = mp.diff(lambda y: Pf(y, p, k), mp.mpf(p) + t, 2)
        if not d2 > 0:
            viol2 += 1
    f018 = Pf(mp.mpf(p) + mp.mpf(9) / 50, p, k)
    viol3 = 0 if f018 > 0 else 1
    tg = [mp.mpf(9) / 50 + (1 - mp.mpf(9) / 50) * j / 400 for j in range(0, 400)] + [1 - mp.mpf(10) ** (-e) for e in range(3, 9)]
    for t in tg:
        if not Pf(mp.mpf(p) + t, p, k) > 0:
            viol3 += 1
    # Vorzeichenwechsel auf Gesamtgitter
    grid = [mp.mpf(0.18) * j / 600 for j in range(1, 601)] + tg
    vals = [Pf(mp.mpf(p) + t, p, k) for t in grid]
    sgn = [v > 0 for v in vals]
    changes = sum(1 for a, b in zip(sgn, sgn[1:]) if a != b) + (0 if not sgn[0] else 1)  # f<0 am linken Rand (S1) -> Wechsel, wenn erster Gitterwert > 0
    # Nullstelle per Bisektion zwischen letztem negativen und erstem positiven Gitterpunkt
    z = None
    fz1 = None
    ok4 = False
    idx = next((j for j, s in enumerate(sgn) if s), None)
    if idx is not None:
        lo = mp.mpf(0) if idx == 0 else grid[idx - 1]
        hi = grid[idx]
        for _ in range(110):
            mid = (lo + hi) / 2
            if Pf(mp.mpf(p) + mid, p, k) > 0:
                hi = mid
            else:
                lo = mid
        z = (lo + hi) / 2
        fz1 = mp.diff(lambda y: Pf(y, p, k), mp.mpf(p) + z, 1)
        ok4 = (changes == 1) and (z < mp.mpf(9) / 50) and (fz1 > 0) and (fz1 > -f0_closed / z)
    ok = ok1 and viol2 == 0 and viol3 == 0 and ok4
    return dict(ok=ok, ok1=ok1, r1=r1, viol2=viol2, viol3=viol3, changes=changes, z=z, fz1=fz1, f0=f0_closed, f018=f018, bound=(-f0_closed / z if z else None))


t0 = time.time()
log("E7829 Zusammenbau-Gegenprobe", time.strftime("%Y-%m-%d %H:%M"))
tot = 0
bad = []
rows = []
for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 101, 499, 997):
    for m in (1.0, 1.7, 3.0, 10.0):
        k = m * 3.25 * (p + 1) * math.log(p)
        if p > 100 and m > 3:
            pass
        c = cell(p, k)
        tot += 1
        rows.append((p, m, k, c))
        if not c["ok"]:
            bad.append((p, m))
        log(f"p={p:4d} m={m:5.1f} kappa={k:10.3f}: ok={c['ok']} f(0)={float(c['f0']):.3e} r1={float(c['r1']):.1e} viol2={c['viol2']} viol3={c['viol3']} Wechsel={c['changes']} z={float(c['z']):.6e} p*z={float(c['z']*p):.4f} f'(z)={float(c['fz1']):.5f} (>{float(c['bound']):.5f}) f(0.18)/p={float(c['f018']/p):.3e}")
log(f"Zellen {tot}, Verletzungen {len(bad)}: {bad}")
log("== R6 Mutanten")
mk = 0.3 * 3.25 * 6 * math.log(5)
c = cell(5, mk)
log(f"Mutant a (p=5, kappa=0,3 kappa_min={mk:.2f}): ok={c['ok']} viol2={c['viol2']} viol3={c['viol3']} Wechsel={c['changes']}")
res_a = not c["ok"]
c2 = cell(9, 3.25 * 10 * math.log(9))
log(f"Mutant b (n=9, zusammengesetzt): ok={c2['ok']} viol2={c2['viol2']} viol3={c2['viol3']} Wechsel={c2['changes']}")
res_b = not c2["ok"]
log("Mutanten scheitern wie erwartet:", res_a, res_b)
log(f"({time.time()-t0:.0f} s)")
with open(os.path.join(HERE, "logs", "E7829_zusammenbau__20261006_1500.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
