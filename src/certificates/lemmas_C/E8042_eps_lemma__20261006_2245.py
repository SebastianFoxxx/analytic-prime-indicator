"""E8042 (T36-F-1260) MESSLAUF: Lemma E (Restschranke eps_p fuer p >= 23, B3 der Gegenlese T35-V-1091).

Gate E8039 gruen, committet vor diesem Lauf.  Objekt (C4): B_rest = sum_{i >= 2, i != p+1} i (phi_i'' f_i + 2 phi_i' f_i') der Zerlegung
P_sigma'' = A + B_edge + B_rest, p prim >= 23, kappa >= C (p+1) ln p, C = 3,25, t in [0, 9/50].

VORAB-REGELN:
 R1  Arb-Wert eps(23) = 1,1387e-4 (Altskript p1198uni.eps_uniform, relativ 1e-12 identisch laut Gate E7), eps(31) = 2,5126e-5; Teilsummen e1 (i <= p) und e2 (i >= p+2)
     werden ausgegeben.  Erwartung: e2 dominiert (e2/e1 > 10).
 R2  Handpruefbare Schwellen des Beweises (Arb): 2 C ln 23 > 2 ; 2/ln 23 < 2 C (41/50)(24/24.18) (Monotonie-Bedingungen); C ln 23 > 1,22 (kappa_min >= 1/m_i);
     Quotient (26/25)^3 exp(-ell 24/24.18) < 1/2 (Summe i >= p+2 <= 2 * erster Term).
 R3  Echte Gegenprobe: |B_rest| (exakt aus den geschlossenen Formen, mpmath dps 30, Fenster i in [max(2,p-70), p+14] (Rest < 1e-300)) <= p * eps(p)
     fuer p in einer Liste von 40 Primzahlen 23..10007, kappa/kappa_min in {1; 1,2; 2; 5}, 16 t-Punkte in [1e-3, 0,18]: 0 Verletzungen;
     kleinster Faktor eps(p) p / |B_rest| wird ausgegeben (Erwartung laut Gegenlese 10^2..10^11).
 R4  Summenschranke: fuer jede der 40 Primzahlen ist sum_{i != p+1} T_i(kappa_min) (f = 1, Formel des Lemmas) <= p eps(p) (die Schranke des Lemmas ist wirklich eine
     Obermenge der Summe der Einzelschranken), und die Teilsumme i >= p+2 ist <= 2 T_{p+2}.
 R5  Mutant: eps mit C = 1,0 statt 3,25 in der Formel UND kappa = 3,25 kappa_min-Auswertung ist kein Test; stattdessen Mutant 'B_rest ohne den Faktor 1/2 im Quotient' ist
     hier nicht noetig. Detektorkontrolle: eps(p) * 1e-6 (zu klein) verletzt R3 > 0 mal.
"""
import os, sys, math, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "T36_F1198"))
import mpmath as mp
from flint import arb
import l1260lib as L
import p1198uni as U

mp.mp.dps = 30
OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)

t0 = time.time()
log("E8042 Lemma E eps_p", time.strftime("%Y-%m-%d %H:%M"))
C = 3.25
ok_all = True
# R1
for p in (23, 29, 31, 37, 101, 1009):
    e, e1, e2 = L.eps_formula(p)
    log(f"R1 p={p}: eps = {float(e.upper()):.6e}  (i<=p: {float(e1.upper()):.3e}, i>=p+2: {float(e2.upper()):.3e}, e2/e1 = {float((e2/e1).mid()):.1f})")
e23 = L.eps_formula(23)[0]
ok_all &= abs(float(e23.upper()) - 1.138741e-4) < 1e-9
# R2
ell23 = 2 * arb(13) / 4 * arb(23).log()
conds = {
 "2 C ln23 > 2": bool(ell23 > 2),
 "2/ln23 < 2C(41/50)(24/24.18)": bool(2 / arb(23).log() < 2 * arb(13) / 4 * arb(41) / 50 * arb(24) / arb("24.18")),
 "C ln23 > 1.22": bool(arb(13) / 4 * arb(23).log() > 1.22),
 "(26/25)^3 exp(-ell 24/24.18) < 1/2": bool((arb(26) / 25) ** 3 * (-ell23 * 24 / arb("24.18")).exp() < 0.5),
}
log(f"R2 ell(23) = {float(ell23.mid()):.4f}; Bedingungen: {conds}; (26/25)^3 exp(..) = {float(((arb(26)/25)**3 * (-ell23*24/arb('24.18')).exp()).upper()):.3e}")
ok_all &= all(conds.values())
# R3
PR = [p for p in range(23, 400) if all(p % q for q in range(2, int(p**0.5) + 1))]
PR = [p for p in PR if p % 6 in (1, 5)] + [401, 601, 1009, 2003, 5003, 10007]
log(f"R3/R4: {len(PR)} Primzahlen")
def parts(p, i, t, kappa):
    p = mp.mpf(p); t = mp.mpf(t); kappa = mp.mpf(kappa)
    X = p + t + 1
    s = 2 * kappa * (1 - i / X); s1 = 2 * kappa * i / X ** 2; s2 = -4 * kappa * i / X ** 3
    e_ = mp.exp(-abs(s)); d1 = e_ / (1 + e_) ** 2; d2 = -d1 * mp.tanh(s / 2)
    D = mp.sin(mp.pi * (p + t) / i); N = mp.sin(mp.pi * t)
    h = N / (i * D); h1 = mp.pi * (mp.cos(mp.pi * t) * D - N * mp.cos(mp.pi * (p + t) / i) / i) / (i * D ** 2)
    return d1 * s1, d2 * s1 ** 2 + d1 * s2, h * h, 2 * h * h1
def Brest(p, t, kappa):
    tot = mp.mpf(0)
    for i in range(max(2, p - 70), p + 15):
        if i == p + 1:
            continue
        ph1, ph2, f, f1 = parts(p, i, t, kappa)
        tot += i * (ph2 * f + 2 * ph1 * f1)
    return tot
tgrid = [1e-3, 5e-3] + [0.18 * k / 14 for k in range(1, 15)]
worst = 0.0; minfac = None; viol = 0; viol_mut = 0; n = 0
for p in PR:
    eps = float(L.eps_formula(p)[0].upper())
    kmin = C * (p + 1) * math.log(p)
    for kf in (1.0, 1.2, 2.0, 5.0):
        for t in tgrid:
            b = abs(Brest(p, t, kf * kmin))
            n += 1
            ratio = float(b) / (p * eps)
            worst = max(worst, ratio)
            if b > p * eps: viol += 1
            if b > p * eps * 1e-6: viol_mut += 1
log(f"R3 {n} Faelle: Verletzungen |B_rest| <= p eps(p): {viol}; max |B_rest|/(p eps) = {worst:.3e} (kleinster Faktor {1/worst if worst>0 else float('inf'):.3e}); Mutant eps*1e-6 verletzt {viol_mut} Faelle")
ok_all &= (viol == 0) and (viol_mut > 0)
# R4
viol4 = 0; viol4b = 0
for p in PR:
    ell = 2 * C * math.log(p)
    kmin = C * (p + 1) * math.log(p)
    Tsum = 0.0; Tfar = 0.0; Tp2 = None
    for i in range(2, p + 400):
        if i == p + 1: continue
        m = (1 - i / (p + 1)) if i <= p else (i / (p + 1.18) - 1)
        z = 2 * kmin * i / (p + 1) ** 2; zp = 4 * kmin * i / (p + 1) ** 3
        T = i * math.exp(-2 * kmin * m) * (z * z + zp + 4 * math.sqrt(2) * math.pi * z)
        Tsum += T
        if i >= p + 2: Tfar += T
        if i == p + 2: Tp2 = T
    eps = float(L.eps_formula(p)[0].upper())
    if Tsum > p * eps * (1 + 1e-12): viol4 += 1
    if Tfar > 2 * Tp2: viol4b += 1
log(f"R4 Summe der Einzelschranken <= p eps(p): Verletzungen {viol4}; Teilsumme i>=p+2 <= 2 T_(p+2): Verletzungen {viol4b}")
ok_all &= viol4 == 0 and viol4b == 0
log("ERGEBNIS:", "GRUEN" if ok_all else "ROT", f"({time.time()-t0:.0f} s)")
open(os.path.join(HERE, "logs", "E8042_eps_lemma__20261006_2245.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
