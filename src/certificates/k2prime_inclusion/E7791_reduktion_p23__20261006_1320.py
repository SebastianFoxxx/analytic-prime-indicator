"""E7791  T36-F-1195 R2: p-uniforme Reduktion fuer alle Primzahlen p >= 23 und Intervallzertifikat (mpmath.iv).

SATZ-KANDIDAT (K2', Teil p >= 23).  Sei p >= 23 prim, kappa >= C (p+1) ln p, t in [0,18; 1).  Dann
    P_sigma(p+t;kappa) >= 0,0015 p - p*Delta(p,t,C)   mit   Delta <= Delta_up(t,C) := (24/23) g_24(1-t) W_C(t) + D0(C),
  wobei  g_n(s) = (sinc(s)/sinc(s/n))^2  (= f_n(n-1+t) mit s = 1-t, sinc(y) = sin(pi y)/(pi y)),
         W_C(t) = exp(-2 C ln 23 * t/(1+t/24)),
         D0(C)  = exp(-a0 (1+t_min)) / (1 - exp(-a0)),  a0 = 2 C ln 23 * 24/25,  t_min = 0,1796875.
  BEWEIS-SKIZZE (Papier, exakte Schritte, im Report ausgefuehrt):
   (i)  P_sigma = P_inf - (1-phi_{p+1}) E - D + H,  H >= 0   (Zerlegung, E7790 gruen)
   (ii) P_inf(p+t) >= 0,0015 p   (K2, T36-F-993-T1, Kap. 17 §17.3; zitiert, nicht neu bewiesen)
   (iii) 1-phi_{p+1}(t) <= exp(-2 kappa t/(p+1+t)) <= exp(-2 C ln p * t/(1+t/(p+1))) <= W_C(t)  (Exponent waechst in p)
   (iv) E/p = ((p+1)/p) g_{p+1}(1-t) <= (24/23) g_24(1-t)   (g_n fallend in n: n sin(pi s/n) waechst in n, tan y >= y)
   (v)  D/p <= sum_{j>=1} exp(-a (j+t)),  a = 2 kappa/(p+1+t) >= a0   (i<=p: f_i <= 1, i/p <= 1, 1-phi_i <= exp(-q_i))
 CERTIFICAT: sup_{t in [t_min,1]} Delta_up(t,C) < 0,0015 mit mpmath.iv (adaptive Bisektion auf dyadischen Teilintervallen
   von [23/128, 1]; sinc ist auf [0,1] fallend -> Intervallenden; Soundness von mpmath.iv ANGENOMMEN, nicht formal geprueft).

VORAB-REGELN:
  R1  Das Zertifikat gelingt fuer C = 6,4 UND fuer C = 3,3; das kleinste zertifizierte C auf dem Raster 0,01 liegt in [3,04; 3,30]
      (Erwartung: knapp ueber 3,039 = Tabellenwert p = 23, weil (iii)-(v) nur wenig verlieren).  FAELLT es auf C > 6,4: Kill-Signal K-C.
  R2  Numerische Kette (Float, p = alle Primzahlen 23 <= p <= 3001 plus 10007, 100003, t-Raster 4001 Punkte in [0,18; 1)):
      0 Verletzungen von  (omega E + D)/p <= Delta_up(t,C)  bei C = 3,3 und C = 6,4  (Gegenbeispiel-Suche zuerst).
  R3  Mutant M1: g_24 durch g_{p+1}(p=1009) ersetzen (falsche Richtung der Monotonie, ist zu klein) -> die Kette R2 faellt an p = 23.
  R4  Mutant M2: W_C mit p = 5 statt 23 in ln -> zu schwach? (ln 5 < ln 23: W groesser) -> Zertifikat gelingt NICHT bei C = 3,3 (muss fallen).
  R5  C = 3,0 darf nicht zertifizierbar sein (Tabelle p = 23: 3,039).
Ausgabe: logs/E7791_reduktion__20261006_1320.txt, csv data/E7791_...csv
"""
import math
import os
import sys
import csv
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1195lib as L

iv = mp.iv
iv.dps = 40
TARGET = 0.0015
TMIN = 23.0 / 128.0
LN23 = None


def sinc_pt(s):
    """sinc fuer Float-Punkt s in [0,1]."""
    if s == 0.0:
        return iv.mpf(1)
    x = iv.mpf(s)
    return iv.sin(iv.pi * x) / (iv.pi * x)


def sinc_int(slo, shi):
    """sinc auf [slo,shi] in [0,1], fallend -> [sinc(shi), sinc(slo)]."""
    a = sinc_pt(shi)
    b = sinc_pt(slo)
    return iv.mpf([a.a, b.b])


def g_int(n, slo, shi):
    A = sinc_int(slo, shi)
    B = sinc_int(slo / n, shi / n)
    r = A / B
    return r * r


def D0(C, q=23):
    a0 = iv.mpf(2 * C) * iv.log(iv.mpf(q)) * iv.mpf(24) / iv.mpf(25)
    return iv.exp(-a0 * (1 + iv.mpf(TMIN))) / (1 - iv.exp(-a0))


def delta_up(C, a, b, nmono=24, q=23, ratio=24.0 / 23.0):
    """Obere Schranke von Delta_up auf t in [a,b] (dyadisch), als iv."""
    tt = iv.mpf([a, b])
    expo = iv.mpf(2 * C) * iv.log(iv.mpf(q)) * tt / (1 + tt / iv.mpf(nmono))
    W = iv.exp(-expo)
    g = g_int(nmono, 1.0 - b, 1.0 - a)
    return iv.mpf(ratio) * g * W


def certify(C, mutant=None, maxdepth=26):
    """True, wenn sup Delta_up(t,C) + D0 < TARGET auf [TMIN,1]; liefert (ok, max_ub, n_intervalle)."""
    q = 23 if mutant != "M2" else 5
    d0 = D0(C, q)
    budget = TARGET - d0.b
    stack = [(TMIN, 1.0, 0)]
    nint = 0
    worst = 0.0
    while stack:
        a, b, dep = stack.pop()
        ub = delta_up(C, a, b, q=q).b
        if ub < budget:
            nint += 1
            worst = max(worst, float(ub))
            continue
        if dep >= maxdepth:
            return False, float(ub), nint
        m = 0.5 * (a + b)
        stack.append((a, m, dep + 1))
        stack.append((m, b, dep + 1))
    return True, worst + float(d0.b), nint


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("E7791 Reduktion p>=23, Intervallzertifikat (mpmath.iv, Soundness angenommen)")
    res = {}
    for C in (3.0, 3.04, 3.05, 3.1, 3.2, 3.3, 6.4):
        ok, w, n = certify(C)
        res[C] = ok
        out(f"  C = {C:5.2f}: zertifiziert = {ok}  (max Delta_up+D0 = {w:.6f}, Ziel {TARGET}; {n} Intervalle)")
    # kleinstes C auf Raster 0,01
    cmin = None
    for k in range(300, 331):
        C = k / 100.0
        if certify(C)[0]:
            cmin = C
            break
    out(f"R1 kleinstes zertifiziertes C (Raster 0,01 ab 3,00): {cmin}; C=6,4: {res[6.4]}, C=3,3: {res[3.3]}")
    r1 = res[6.4] and res[3.3] and cmin is not None and 3.04 <= cmin <= 3.30
    out(f"R1 {'ok' if r1 else 'ROT'}")
    r5 = not res[3.0]
    out(f"R5 C=3,0 nicht zertifizierbar: {'ok' if r5 else 'ROT'}")
    # R4 Mutant M2
    m2 = certify(3.3, mutant="M2")[0]
    out(f"R4 Mutant M2 (ln 5 statt ln 23) bei C=3,3 zertifizierbar? {m2} -> {'ok (faellt wie vorgesehen)' if not m2 else 'ROT'}")
    # R2 numerische Kette
    PR = [p for p in L.primes_upto(3001) if p >= 23] + [10007, 100003]
    tg = np.linspace(TMIN, 1.0 - 1e-9, 4001)
    viol = {3.3: 0, 6.4: 0}
    maxratio = {3.3: 0.0, 6.4: 0.0}
    rows = []
    for C in (3.3, 6.4):
        lnp = math.log(23.0)
        d0 = float(D0(C).b)
        # Delta_up punktweise (Float, mit iv-Auswertung an den Punkten waere teuer; Float mit Vorsicht, nur Detektion)
        s = 1.0 - tg
        g24 = (L.sinc(s) / L.sinc(s / 24.0)) ** 2
        W = np.exp(-2 * C * lnp * tg / (1 + tg / 24.0))
        dup = (24.0 / 23.0) * g24 * W + d0
        for p in PR:
            if p > 3001 and C == 3.3:
                pass
            kap = C * (p + 1) * math.log(p)
            if p <= 3001:
                tgl = tg
            else:
                tgl = tg[::40]
            d = L.parts(p, tgl, kap)
            lhs = (d["om"] * d["E"] + d["D"]) / p
            dd = dup if p <= 3001 else np.interp(tgl, tg, dup)
            bad = int(np.sum(lhs > dd * (1 + 1e-12)))
            viol[C] += bad
            maxratio[C] = max(maxratio[C], float(np.max(lhs / dd)))
            if p in (23, 29, 101, 1009, 3001, 10007, 100003):
                rows.append((C, p, float(np.max(lhs)), float(np.max(lhs / dd))))
        out(f"R2 C={C}: Verletzungen (omega E + D)/p <= Delta_up: {viol[C]} ueber {len(PR)} Primzahlen; max lhs/Delta_up = {maxratio[C]:.4f}")
    r2 = viol[3.3] == 0 and viol[6.4] == 0
    out(f"R2 {'ok' if r2 else 'ROT'}")
    # R3 Mutant M1: g_{1009} statt g_24 -> Kette faellt an p = 23 (bei C=3,3, lhs groesser als dup_M1)
    C = 3.3
    s = 1.0 - tg
    g1009 = (L.sinc(s) / L.sinc(s / 1009.0)) ** 2
    dupM = (24.0 / 23.0) * g1009 * np.exp(-2 * C * math.log(23.0) * tg / (1 + tg / 24.0)) + float(D0(C).b)
    d = L.parts(23, tg, C * 24 * math.log(23))
    lhs = (d["om"] * d["E"] + d["D"]) / 23
    badM = int(np.sum(lhs > dupM))
    out(f"R3 Mutant M1 (g_1009 statt g_24): Verletzungen an p=23: {badM} -> {'ok (faellt)' if badM > 0 else 'ROT (Detektor blind)'}")
    r3 = badM > 0
    gate = r1 and r2 and r3 and r5 and (not m2)
    out("ERGEBNIS: " + ("GRUEN" if gate else "ROT") + f"; kleinstes zertifiziertes C fuer p>=23: {cmin}")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "E7791_reduktion__20261006_1320.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "data", "E7791_reduktion__20261006_1320.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["C", "p", "max_lhs", "max_lhs_over_delta_up"])
        w.writerows(rows)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
