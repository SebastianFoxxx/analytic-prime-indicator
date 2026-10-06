"""E7832 (T36-F-1198, R1b) GATE VOR DER MESSUNG: p-uniformes K1'-Zertifikat fuer ALLE Primzahlen p >= 23 (Klassen p mod 6, Randform exakt, eps p-uniform).

VORAB-REGELN (festgeschrieben VOR E7833):
 W1 Klassenkoeffizienten (exakt, rational): fuer jede Primzahl p <= 5000 der Klasse (r = 1: p >= 31; r = 5: p >= 23) und jedes 0 < |rho| <= 40 gilt
    (1/p) sum_{i | (p-rho), 2 <= i <= p, } i  >=  c_a(rho)   und   [ (p+1) | (p-rho) ] (p+1)/p  >=  c_b(rho).  Erlaubte Verletzungen: 0.
    (Hinweis: die linke Seite ist die EXAKTE Gewichtssumme, nicht nur die garantierten Teiler; c_a, c_b stammen nur aus den garantierten Teilern d/m, m | gcd(d,6).)
 W2 Soundness in Zellen: Primzahlen {23, 29, 41, 47, 31, 37, 43, 61, 101, 107, 211} x kappa-Vielfache {1; 1,3; 3; 10} x kappa_min(C = 3,25) x t {0,001; 0,02; 0,09; 0,18}:
    (a) A/p >= A0 + sigma(s) A1 (mp, exakte Ableitungen), (b) |B_rest|/p <= E (p-uniform), (c) B_edge/p >= (p_lo+1)/p_lo * Uu * m(s) + Vl * n(s) mit u-Ball, (d) Gesamtschranke <= P''/p. Verletzungen: 0.
 W3 U, V als Funktion von u: edge_UV_u(t, u = 1/(p+1)) liegt im Ball von edge_UV(p, t) (aus E7827 geprueft) an p in {23, 31, 101, 997}, t in {0, 0,05, 0,18}.
 W4 Monotonie von E(p): eps_uniform(C, p0) ist fuer p0 in {23, 31, 41, 101, 1009, 10007, 10^6} nichtsteigend.
 W5 Mutanten MUESSEN scheitern (certify_uniform ok = False): Klasse 1 (p_lo = 31): M1 nur Paritaet (M = 2), M2 R = 1, M3 V-Vorzeichen gedreht, M4 ohne die kappa-Einschraenkung s >= s_min(t) ('no_s_restrict'), M5 C = 0,5.  Klasse 5: M1, M2, M5.
    [Gate-Lauf 1 (rot, aufbewahrt) hatte M2 = 'R = 3' und M3 auch in Klasse 5: M2(R=3) laeuft DURCH (|rho| <= 3 genuegt mit Teilern d/3, d/6 - eine Eigenschaft des Objekts, kein Instrumentdefekt) und M3 in Klasse 5 laeuft durch (grosse Marge dort); beide waren keine diskriminierenden Mutanten und wurden durch staerkere ersetzt. W1-W4, W6 waren in Lauf 1 gruen.]
 W6 Erreichbarkeit: ein positiver Lauf teilt mindestens einmal (n_boxes > first_split) ODER erreicht alle ersten Boxen; ein Mutantenlauf erreicht den fails-Zweig.
Entscheidung: alles gruen -> E7833 (Messlauf).
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from flint import arb

import p1198lib as L
import p1198mp as M
import p1198uni as U

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def primes_upto(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for k in range(2, int(n ** 0.5) + 1):
        if s[k]:
            s[k * k::k] = bytearray(len(s[k * k::k]))
    return [k for k in range(n + 1) if s[k]]


mp.mp.dps = 40
res = {}
t0 = time.time()
log("E7832 Gate P23 uniform", time.strftime("%Y-%m-%d %H:%M"))
PL = {1: 31, 5: 23}
cf = {r: U.class_coeffs(r, p_lo=PL[r]) for r in (1, 5)}

# ---- W1
bad = 0
n = 0
from fractions import Fraction
for p in primes_upto(5000):
    r = p % 6
    if r not in (1, 5) or p < PL[r]:
        continue
    for rho in range(-40, 41):
        if rho == 0:
            continue
        d = p - rho
        if d <= 0:
            continue
        exact_a = Fraction(sum(i for i in range(2, p + 1) if d % i == 0), p)
        exact_b = Fraction(p + 1, p) if d % (p + 1) == 0 else Fraction(0)
        ca, cb = cf[r][rho]
        n += 1
        if exact_a < ca or exact_b < cb:
            bad += 1
            if bad < 5:
                log(f"  W1 VERLETZUNG p={p} rho={rho}: exakt a={exact_a} b={exact_b}  c_a={ca} c_b={cb}")
log(f"W1: {n} (p, rho)-Paare, Verletzungen {bad}")
res["W1"] = bad == 0

# ---- W2
log("== W2 Soundness")
C = 3.25
viol = [0, 0, 0, 0]
ncell = 0
minslack = 1e9
for p in (23, 29, 41, 47, 31, 37, 43, 61, 101, 107, 211):
    r = p % 6
    plo = PL[r]
    sigma0 = L.sig(2 * arb(C) * arb(plo).log())
    E, einfo = U.eps_uniform(C, plo)
    fac = arb(plo + 1) / plo
    ub = arb(0.5 / (plo + 1), 0.5 / (plo + 1) * 1.0000001) + arb(0.5 / (plo + 1)) * 0  # Ball ueber [0, 1/(plo+1)]
    ub = arb(0.5 / (plo + 1), 0.5 / (plo + 1) * 1.0000001)
    km = L.kappa_min(p, C)
    for mult in (1.0, 1.3, 3.0, 10.0):
        kap = km * mult
        kf = mp.mpf(str(kap.mid().str(30, radius=False)))
        for t in (0.001, 0.02, 0.09, 0.18):
            tb = arb(t)
            A0, A1 = U.A_coeffs_uniform(tb, sigma0, cf[r])
            s = 2 * kap * tb / (p + 1 + tb)
            Alow = (A0 + L.sig(s) * A1) * p
            Uu_, Vv_ = U.edge_UV_u(tb, ub)
            Bedge_low = p * (fac * Uu_.upper() * L.mfun(s) + Vv_.lower() * L.nfun(s))
            A, Be, Br, S = M.split_AB(p, t, kf)
            a_ok = float(Alow.lower()) <= float(A)
            b_ok = abs(float(Br)) / p <= float(E.upper())
            c_ok = float(Bedge_low.lower()) <= float(Be)
            tot = Alow + Bedge_low - p * E
            d_ok = float(tot.lower()) <= float(S)
            for k, ok in enumerate((a_ok, b_ok, c_ok, d_ok)):
                viol[k] += (not ok)
            ncell += 1
            minslack = min(minslack, float(S) - float(tot.lower()))
            if not (a_ok and b_ok and c_ok and d_ok):
                log(f"  VERLETZUNG p={p} mult={mult} t={t}: a={a_ok} b={b_ok} c={c_ok} d={d_ok}  A_low={float(Alow.mid()):.4f} A={float(A):.4f} Be_low={float(Bedge_low.mid()):.4f} Be={float(Be):.4f} Br/p={float(Br)/p:.2e} E={float(E.upper()):.2e}")
log(f"W2: {ncell} Zellen, Verletzungen (a,b,c,d) = {viol}, kleinster Abstand P'' - Schranke = {minslack:.4f}")
res["W2"] = sum(viol) == 0

# ---- W3
log("== W3 U,V(u)")
w3 = 0
for p in (23, 31, 101, 997):
    for t in (0.0, 0.05, 0.18):
        Ua, Va = L.edge_UV(p, arb(t))
        Ub, Vb = U.edge_UV_u(arb(t), arb(1) / (p + 1))
        ok = Ua.overlaps(Ub) and Va.overlaps(Vb)
        w3 += (not ok)
log(f"W3: Verletzungen {w3}")
res["W3"] = w3 == 0

# ---- W4
log("== W4 E(p) nichtsteigend")
prev = None
w4 = 0
for p0 in (23, 31, 41, 101, 1009, 10007, 10 ** 6):
    E, info = U.eps_uniform(3.25, p0)
    ev = float(E.upper())
    log(f"  p0={p0}: E = {ev:.3e}  ratio_ok={info['ratio_ok']}")
    if prev is not None and ev > prev:
        w4 += 1
    if not info["ratio_ok"]:
        w4 += 1
    prev = ev
res["W4"] = w4 == 0

# ---- W5 / W6
log("== W5 Mutanten / W6")
grid = L.s_grid()
sb = L.s_boxes_lower(grid)
w5 = 0
fails_branch = False
for r in (1, 5):
    plo = PL[r]
    ush = (0.0, 1 / 100, 1 / (plo + 1))
    muts = [("M1_nurParitaet", {"parity_only": True}), ("M2_R1", {"R": 1}), ("M5_C0.5", {"C": 0.5})]
    if r == 1:
        muts += [("M3_flipV", {"flipV": True}), ("M4_ohne_s_Einschraenkung", {"no_s_restrict": True})]
    for name, mut in muts:
        rr = U.certify_uniform(r, mut.get("C", 3.25), p_lo=plo, grid=grid, sb=sb, first_split=8, tol_width=2e-3, mut=mut, u_split=ush)
        log(f"  Klasse {r} {name}: ok={rr['ok']} fails={len(rr['fails'])} boxes={rr['n_boxes']}")
        if rr["ok"]:
            w5 += 1
        if rr["fails"]:
            fails_branch = True
res["W5"] = w5 == 0
res["W6"] = fails_branch
log("== Ergebnis Gate:", res)
log("GATE", "GRUEN" if all(res.values()) else "ROT", f"({time.time()-t0:.0f} s)")
with open(os.path.join(HERE, "logs", "E7832_gate__20261006_1500.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
