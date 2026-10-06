"""E8039 (T36-F-1260) GATE VOR DER MESSUNG: Detektoren fuer die zwei Hilfsabschaetzungen (Klassengewichte, Restschranke eps_p).

Dieses Skript wird committet, BEVOR die Messlaeufe E8040-E8044 gefahren werden.

VORAB-REGELN (Erwartungen, bekannte Antworten):
 K1  Formeln des Klassen-Lemmas c_a(q,rho) = (1-rho/q) K_r(rho), c_b(q,-1) = 1+1/q (sonst 0) gelten exakt fuer ALLE q = r (6) in [53, 3000],
     |rho| <= 40, r in {1,5}: 0 Verletzungen (Definition vs. Formel, Bruchrechnung).                 [GRUEN erwartet]
 K2  Mutant: Schwelle q > R+12 gesenkt auf q >= 36 -> Verletzungen > 0 (z. B. q = 41, rho = 35: d = 6, d/6 = 1 < 2).   [MUSS fallen]
 K3  Mutant: K um 1 % vergroessert -> Verletzungen > 0.                                                 [MUSS fallen]
 K4  Mutant: Regel 'm >= 2 oder rho >= 1' ignoriert (m = 1 immer gezaehlt) -> Verletzungen > 0.         [MUSS fallen]
 K5  Neue exakte Tabelle <= alte Tabelle (Stellvertreter q = 10^9+7) eintragsweise, Abstand <= 1e-6.      [GRUEN erwartet]
 E0  Geschlossene Formen fuer phi', phi'', f', f'' stimmen mit mp.diff ueberein (relativ 1e-12) an 24 Punkten.   [GRUEN erwartet]
 E1  Bernstein: |f_i'| <= 2 pi sqrt(f_i) fuer 20000 Zufallspaare (i <= 3000, x in (0,3000)): 0 Verletzungen.   [GRUEN erwartet]
 E2  Mutant: Konstante 2 statt 2 pi (wahres Supremum von |f'|/sqrt f liegt bei ca. 2,74) -> Verletzungen > 0.                                               [MUSS fallen]
 E3  Termschranke |i(phi'' f + 2 phi' f')| <= i e^{-|s|}[(s'^2+|s''|) f + 4 pi s' sqrt f]: 0 Verletzungen (12000 Zufallsfaelle). [GRUEN erwartet]
 E4  Mutant: Kreuzterm-Konstante 4 pi -> pi/2 in der Schranke -> Verletzungen > 0.                                        [MUSS fallen]
 E5  Monotonie der gleichmaessigen Termschranke T_i(kappa) fuer kappa >= kappa_min (C = 3,25): 0 Verletzungen; Mutant C = 0,2 (kappa_min < 1/m_i) -> Verletzungen > 0.
 E6  eps(p) (eigene Implementierung) faellt entlang p = 23 * 1,01^k (k <= 3000): 0 Verletzungen; Mutant (41/50 -> 1/1000 im Exponenten) -> steigt.
 E7  eps(23) eigene Implementierung == p1198uni.eps_uniform(3,25, 23) (relativ 1e-12).                     [GRUEN erwartet]
Erfolg des Gates: alle GRUEN-Zeilen gruen UND alle MUSS-FALLEN-Zeilen gefallen.
"""
import math
import os
import random
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "T36_F1198"))
import mpmath as mp
from flint import arb
import l1260lib as L

mp.mp.dps = 40
OUT = []
RES = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def verdict(name, ok, expect_ok, note=""):
    good = (ok == expect_ok)
    RES[name] = good
    log(f"{name}: gemessen {'gruen' if ok else 'ROT/gefallen'}; erwartet {'gruen' if expect_ok else 'MUSS fallen'} -> {'OK' if good else 'GATE-FEHLER'} {note}")


# ------------------------------------------------------------------ K1-K5
bad1 = sum(L.check_structure(r, 53, 3000)[0] for r in (1, 5))
verdict("K1", bad1 == 0, True, f"(Verletzungen {bad1})")
bad2 = [L.check_structure(r, 36, 3000) for r in (1, 5)]
verdict("K2", all(b[0] == 0 for b in bad2), False, f"(Verletzungen {[b[0] for b in bad2]}, erste {[b[2] for b in bad2]})")
bad3 = [L.check_structure(r, 53, 600, K_scale=Fraction(101, 100))[0] for r in (1, 5)]
verdict("K3", all(b == 0 for b in bad3), False, f"(Verletzungen {bad3})")
bad4 = 0
for r in (1, 5):
    for q in range(53, 600):
        if q % 6 != r:
            continue
        for rho in range(-40, 41):
            if rho == 0:
                continue
            ca_f = Fraction(q - rho, q) * L.K_formula(r, rho, threshold_m1=False)
            if L.c_a_q(q, rho) != ca_f:
                bad4 += 1
verdict("K4", bad4 == 0, False, f"(Verletzungen {bad4})")
import p1198uni as U
okK5 = True
maxd = 0.0
for r, plo in ((1, 31), (5, 23)):
    old = U.class_coeffs(r, p_lo=plo)
    new = L.class_coeffs_exact(r, plo)
    for rho in old:
        for k in (0, 1):
            d = old[rho][k] - new[rho][k]
            if d < 0 or d > Fraction(1, 10 ** 6):
                okK5 = False
            maxd = max(maxd, float(d))
verdict("K5", okK5, True, f"(max Differenz alt-neu {maxd:.3e})")

# ------------------------------------------------------------------ E0: geschlossene Formen
def parts(p, i, t, kappa):
    p = mp.mpf(p); t = mp.mpf(t); kappa = mp.mpf(kappa)
    X = p + t + 1
    s = 2 * kappa * (1 - i / X)
    s1 = 2 * kappa * i / X ** 2
    s2 = -4 * kappa * i / X ** 3
    sg = 1 / (1 + mp.exp(-s))
    e_ = mp.exp(-abs(s))
    d1 = e_ / (1 + e_) ** 2                  # sigma'(s), stabil (kein 1 - sigma bei sigma ~ 1)
    d2 = -d1 * mp.tanh(s / 2)                # sigma''(s) = sigma' (1 - 2 sigma) = -sigma' tanh(s/2)
    phi1 = d1 * s1
    phi2 = d2 * s1 ** 2 + d1 * s2
    D = mp.sin(mp.pi * (p + t) / i)
    N = mp.sin(mp.pi * t)
    h = N / (i * D)
    h1 = mp.pi * (mp.cos(mp.pi * t) * D - N * mp.cos(mp.pi * (p + t) / i) / i) / (i * D ** 2)
    f = h * h
    f1 = 2 * h * h1
    return dict(s=s, s1=s1, s2=s2, phi1=phi1, phi2=phi2, f=f, f1=f1, phi=sg)


random.seed(8039)
okE0 = True
mp.mp.dps = 120
for _ in range(24):
    p = random.choice([23, 29, 31, 37, 101, 211])
    i = random.randint(2, p + 6)
    if i == p + 1:
        i = p + 2
    t = random.uniform(0.01, 0.18)
    kappa = random.uniform(0.3, 1.5) * (p + 1)   # moderate s: mp.diff verliert bei phi ~ 1 - e^{-93} die relative Genauigkeit (Instrumentgrenze, C8)
    P = parts(p, i, t, kappa)
    sgn = 1 if P["s"] < 0 else -1            # s > 0: phi' = -(1-phi)' mit 1-phi = 1/(1+e^{s}) (kein 1-minus-winzig)
    phi = lambda tt: (1 / (1 + mp.exp(-2 * mp.mpf(kappa) * (1 - i / (p + tt + 1))))) if sgn == 1 else -1 / (1 + mp.exp(2 * mp.mpf(kappa) * (1 - i / (p + tt + 1))))
    ff = lambda tt: (mp.sin(mp.pi * tt) / (i * mp.sin(mp.pi * (p + tt) / i))) ** 2
    nd1 = mp.diff(phi, mp.mpf(t), 1)
    nd2 = mp.diff(phi, mp.mpf(t), 2)
    nf1 = mp.diff(ff, mp.mpf(t), 1)
    for a, b in ((P["phi1"], nd1), (P["phi2"], nd2), (P["f1"], nf1)):
        if abs(a - b) > 1e-12 * max(abs(a), abs(b), mp.mpf(10) ** -300) + mp.mpf(10) ** -280:
            okE0 = False
mp.mp.dps = 25
verdict("E0", okE0, True)

# ------------------------------------------------------------------ E1/E2 Bernstein
def bern_viol(const, n=20000, seed=1):
    rr = random.Random(seed)
    v = 0
    worst = 0.0
    for _ in range(n):
        i = rr.randint(2, 3000)
        x = rr.uniform(0.0, 3000.0)
        if rr.random() < 0.3:
            x = rr.randint(0, 100) + rr.choice([1e-3, 0.05, 0.18, 0.5, 0.9, 0.999])
        xm = mp.mpf(x)
        D = mp.sin(mp.pi * xm / i)
        if abs(D) < 1e-12:
            continue
        N = mp.sin(mp.pi * xm)
        h = N / (i * D)
        h1 = mp.pi * (mp.cos(mp.pi * xm) * D - N * mp.cos(mp.pi * xm / i) / i) / (i * D ** 2)
        f = h * h
        f1 = abs(2 * h * h1)
        bound = const * mp.sqrt(f)
        worst = max(worst, float(f1 / bound) if bound > 0 else 0.0)
        if f1 > bound * (1 + mp.mpf(10) ** -15):
            v += 1
    return v, worst


mp.mp.dps = 25
v1, w1 = bern_viol(2 * mp.pi, 6000)
verdict("E1", v1 == 0, True, f"(Verletzungen {v1}, max |f'|/(2 pi sqrt f) = {w1:.4f})")
v2, w2 = bern_viol(2, 6000)
verdict("E2", v2 == 0, False, f"(Verletzungen {v2}, max Verhaeltnis {w2:.4f})")

# ------------------------------------------------------------------ E3/E4 Termschranke
def term_viol(n, drop_s2=False, seed=3, cross=None):
    rr = random.Random(seed)
    v = 0
    worst = 0.0
    for _ in range(n):
        p = rr.choice([23, 29, 31, 37, 41, 47, 53, 101, 307, 1009])
        C = rr.choice([3.25, 3.25, 4.0, 6.5, 12.0])
        kappa = C * (p + 1) * math.log(p) * rr.choice([1.0, 1.0, 1.3, 2.0])
        i = rr.choice(list(range(max(2, p - 8), p)) + list(range(p + 2, p + 9)) + [rr.randint(2, p)])
        t = rr.uniform(0.0005, 0.18)
        P = parts(p, i, t, kappa)
        term = abs(i * (P["phi2"] * P["f"] + 2 * P["phi1"] * P["f1"]))
        s = abs(P["s"])
        s2 = 0 if drop_s2 else abs(P["s2"])
        bound = i * mp.exp(-s) * ((P["s1"] ** 2 + s2) * P["f"] + (4 * mp.pi if cross is None else cross) * P["s1"] * mp.sqrt(P["f"]))
        if term > bound * (1 + mp.mpf(10) ** -15):
            v += 1
        worst = max(worst, float(term / bound) if bound > 0 else 0)
    return v, worst


v3, w3 = term_viol(1500)
verdict("E3", v3 == 0, True, f"(Verletzungen {v3}, max Verhaeltnis {w3:.4f})")
v4, w4 = term_viol(1500, cross=mp.pi / 2)
verdict("E4", v4 == 0, False, f"(Verletzungen {v4}, max Verhaeltnis {w4:.4f})")

# ------------------------------------------------------------------ E5 Monotonie T_i(kappa)
def mono_viol(C, p_list=(23, 31, 101, 1009)):
    """T_i(kappa) = i e^{-2 kappa m_i}[(z^2 + z') + 4 sqrt2 pi z], z = 2 kappa i/(p+1)^2, z' = 4 kappa i/(p+1)^3, m_i wie im Lemma (f = 1).
    Zaehlt (p,i,kappa-Paare) mit T_i(1,05 kappa) > T_i(kappa) fuer kappa >= kappa_min."""
    v = 0
    for p in p_list:
        kmin = C * (p + 1) * math.log(p)
        for i in list(range(2, p + 1, max(1, p // 40))) + [p] + list(range(p + 2, p + 40)):
            m = (1 - i / (p + 1)) if i <= p else (i / (p + 1.18) - 1)
            for k in (1.0, 1.5, 2.0, 5.0):
                for kk in (kmin * k, kmin * k * 1.05):
                    pass
                T = lambda kap: i * math.exp(-2 * kap * m) * ((2 * kap * i / (p + 1) ** 2) ** 2 + 4 * kap * i / (p + 1) ** 3 + 4 * math.sqrt(2) * math.pi * 2 * kap * i / (p + 1) ** 2)
                if T(kmin * k * 1.05) > T(kmin * k) * (1 + 1e-12):
                    v += 1
    return v


v5 = mono_viol(3.25)
verdict("E5", v5 == 0, True, f"(Verletzungen {v5})")
v5m = mono_viol(0.2)
verdict("E5-Mutant", v5m == 0, False, f"(Verletzungen {v5m})")

# ------------------------------------------------------------------ E6 / E7 eps(p)
def eps_decr_viol(expo_coeff=None, kmax=3000):
    v = 0
    prev = None
    p = Fraction(23)
    for k in range(kmax):
        pp = 23 * (1.01 ** k)
        if expo_coeff is None:
            e, _, _ = L.eps_formula(pp)
        else:
            e = mut_eps(pp, expo_coeff)
        val = float(e.mid())
        if prev is not None and val > prev * (1 + 1e-12):
            v += 1
        prev = val
    return v


def mut_eps(p, coeff):
    p = arb(p)
    C = arb(13) / 4
    ell = 2 * C * p.log()
    c = 4 * arb(2).sqrt() * L.PI
    Q1 = ell * ell + c * ell + 2 * ell / (p + 1)
    z1 = ell * (p + 2) / (p + 1)
    Q2 = z1 * z1 + 2 * ell * (p + 2) / ((p + 1) ** 2) + c * z1
    expo = ell * (p + 1) * arb(coeff) / (p + 1 + arb(9) / 50)
    return (-ell).exp() * Q1 / (1 - (-ell).exp()) + 2 * (p + 2) / p * (-expo).exp() * Q2


v6 = eps_decr_viol()
verdict("E6", v6 == 0, True, f"(Verletzungen {v6})")
v6m = eps_decr_viol(expo_coeff=0.001)
verdict("E6-Mutant", v6m == 0, False, f"(Verletzungen {v6m})")
e23 = L.eps_formula(23)[0]
for plo in (23, 31):
    ref, _info = U.eps_uniform(3.25, plo)
    mine = L.eps_formula(plo)[0]
    rel = abs(float(mine.mid()) - float(ref.mid())) / float(ref.mid())
    verdict(f"E7(p={plo})", rel < 1e-12, True, f"(eigen {float(mine.upper()):.6e}, p1198uni {float(ref.upper()):.6e}, rel {rel:.2e})")

allgood = all(RES.values())
log("GATE:", "GRUEN" if allgood else "ROT", f"({sum(RES.values())}/{len(RES)})")
with open(os.path.join(HERE, "logs", "E8039_gate__20261006_2200.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
sys.exit(0 if allgood else 1)
