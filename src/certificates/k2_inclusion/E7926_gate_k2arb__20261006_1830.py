# -*- coding: utf-8 -*-
"""E7926 - GATE VOR DEN MESSLAEUFEN (Loop T36-F-1251, sub-f1251).
Vorab festgelegte Pass-Regeln: GATE_E7926_E7933.md (im selben Commit).
Prueft die ARB-Instrumente an Faellen mit bekannter Antwort und mit Mutanten, BEVOR das K2-Zertifikat
(E7927) und die Kette (E7929) laufen. Eigener Code (k2lib.py), kein Import aus T36_F993 / T35_V945.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2lib import *
from flint import arb, fmpq, ctx
import mpmath as mp
mp.mp.dps = 60   # (Lauf 1 rot: mpf(a)/b wurde vor dem ersten dps-Setzen mit 53 Bit gebildet - Instrumentfehler C8)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "E7926_gate_k2arb__20261006_1830.txt")
lines = []
def out(s=""):
    print(s); lines.append(s)

res = {}
def check(name, cond, info=""):
    res[name] = bool(cond)
    out(f"[{'OK ' if cond else 'ROT'}] {name} {info}")

def tq(a, b):  # exakte rationale t als arb
    return arb(fmpq(a, b))

# ---------------------------------------------------------------- G1 Evaluator gegen mpmath (separat geschrieben)
def L_mp(R, p, t, cdict=None):
    mp.mp.dps = 60
    t = mp.mpf(t); p = mp.mpf(p)
    pi = mp.pi
    def c(rho):
        if rho == -1: return mp.mpf(1) / 2
        if rho >= 1: return mp.mpf(1) / 2 if rho % 2 else mp.mpf(0)
        return -mp.mpf(1) / 2 if rho % 2 else mp.mpf(-1)
    def gg(rho): return (mp.sin(pi * t) / (pi * (rho + t))) ** 2
    S = mp.mpf(0); S1 = mp.mpf(0)
    for rho in range(-R, R + 1):
        if rho == 0: continue
        S += c(rho) * gg(rho)
        if rho >= 1: S1 += rho * gg(rho)
    T = (mp.sin(pi * t) / pi) ** 2 * 2 * R / (R * R - t * t)
    return S - mp.mpf(3) / (2 * p) * S1 - T - t / p

maxdiff = 0
for (a, b) in [(18, 100), (3, 10), (1, 2), (9, 10), (99, 100), (999, 1000)]:
    v = L(19, arb(23), tq(a, b))
    d = abs(mp.mpf(v.mid().str(50, radius=False)) - L_mp(19, 23, mp.mpf(a) / b))
    maxdiff = max(maxdiff, float(d))
    out(f"  G1 t={a}/{b}: Arb {v.str(18)}  |Arb-mpmath|={float(d):.2e}")
check("G1_arb_gegen_mpmath", maxdiff < 1e-30, f"max Diff {maxdiff:.2e}")

# ---------------------------------------------------------------- G2 bekannte Antworten (T35-V-945/946, nur als Eichwerte)
known = {(18, 100): 0.0018456269, (3, 10): 0.019982, (1, 2): 0.109040, (9, 10): 0.432708}
ok = True
for (a, b), v0 in known.items():
    v = L(19, arb(23), tq(a, b))
    dd = abs(float(v.mid()) - v0)
    out(f"  G2 t={a}/{b}: {v.str(12)} gegen bekannt {v0}  (Diff {dd:.1e})")
    ok &= dd < 2e-6
check("G2_bekannte_antworten", ok)
v = L(15, arb(19), tq(18, 100))
check("G2b_R15_p19_negativ_-0,001117", abs(float(v.mid()) + 0.001117) < 2e-6, f"{v.str(10)}")

# ---------------------------------------------------------------- G3 Poisson-Identitaet (Star) und Mutanten
def Pinf_direct(p, tq_):
    t = arb(tq_)
    st2 = t.sin_pi() ** 2
    s = arb(0)
    for i in range(2, p + 2):
        z = (arb(p) + t) / i
        s += st2 / (i * z.sin_pi() ** 2) if False else st2 / (i * ((arb(p) + t) / i).sin_pi() ** 2)
    return s - (arb(p) + t)

def weight(p, rho):
    d = p - rho
    if d == 0:
        return (p + 1) * (p + 2) // 2 - 1
    d = abs(d)
    return sum(i for i in range(2, p + 2) if d % i == 0)

def poisson_float(p, t, N, drop_t=False, w0_zero=False):
    s = 0.0
    for rho in range(-N, N + 1):
        w = weight(p, rho)
        if w0_zero and rho == 0: w = 0
        gr = (math.sin(math.pi * t) / (math.pi * (rho + t))) ** 2
        s += (w - p) * gr
    return s - (0 if drop_t else t)

p0, t0f = 23, fmpq(3, 10)
Pd = Pinf_direct(p0, t0f)
N = 20000
Ps = poisson_float(p0, 0.3, N)
dif = abs(float(Pd.mid()) - Ps)
out(f"  G3 P_inf(23,3/10) direkt (Arb) {Pd.str(15)}, Poisson N={N}: {Ps:.12f}, Diff {dif:.2e}, Schranke 6p/N={6*p0/N:.2e}")
check("G3_poisson_identitaet", dif < 6 * p0 / N)
dif_m1 = abs(float(Pd.mid()) - poisson_float(p0, 0.3, N, drop_t=True))
dif_m2 = abs(float(Pd.mid()) - poisson_float(p0, 0.3, N, w0_zero=True))
out(f"  G3 Mutanten: ohne -t: Diff {dif_m1:.3f}; w_p(0):=0: Diff {dif_m2:.3f}")
check("G3_mutanten_fallen", dif_m1 > 100 * dif and dif_m2 > 100 * dif)

# ---------------------------------------------------------------- G4 Lemma 2 (Gewichtstabelle) an kleinen p, Negativkontrolle
def w_under(p, rho):
    d = p - rho
    if rho == 0: return p
    if rho == -1: return (p + 1) + (p + 1) // 2
    if rho >= 1: return d + d // 2 if rho % 2 else d
    # rho <= -2
    return (p + abs(rho)) // 2 if rho % 2 else 0

viol = 0; cells = 0; viol_ctrl = 0
for p in [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
    for rho in range(-19, 20):
        cells += 1
        w = weight(p, rho)
        if w < w_under(p, rho): viol += 1
        # Negativkontrolle: erfundenes d/3 ohne 3|d  (Gewicht d + d/2 + d//3 fuer ungerade d.. immer)
        d = p - rho
        if rho >= 1 and w < d + d // 2 + d // 3: viol_ctrl += 1
check("G4_lemma2_gewichte", viol == 0, f"{cells} Zellen, {viol} Verletzungen; Negativkontrolle (erfundenes d/3) {viol_ctrl} Verletzungen")
check("G4_negativkontrolle_faellt", viol_ctrl > 50)

# ---------------------------------------------------------------- G5 Gegenfamilien (muessen das Zertifikat NICHT bestehen)
def c_only_d(rho):
    # nur Teiler d (ohne d/2): (w-p)/p im Limes p->inf
    if rho == -1: return arb(0)       # w >= p+1
    if rho >= 1: return arb(0)        # w >= d = p-rho -> c = 0 (Limes)
    return arb(-1)                    # rho <= -2: d > p+1 -> kein garantierter Teiler
v = L(19, arb(23), tq(18, 100), cfun=c_only_d)
check("G5_nur_teiler_d_negativ", (v < 0), f"L={v.str(10)}")
def c_none(rho): return arb(-1) if rho != 0 else arb(0)
v = L(19, arb(23), tq(18, 100), cfun=c_none)
check("G5_ohne_teilerinput_negativ", (v < 0), f"L={v.str(10)}")

# ---------------------------------------------------------------- G6 Mutanten M1/M2 verschieben den Evaluator
def c_M1(rho): return arb(1) if rho == -1 else cinf(rho)
def c_M2(rho): return arb(1) / 2 if (rho % 2 != 0) else cinf(rho)
base = L(19, arb(23), tq(18, 100))
m1 = L(19, arb(23), tq(18, 100), cfun=c_M1)
m2 = L(19, arb(23), tq(18, 100), cfun=c_M2)
# Korrektur nach Lauf 1 (rot): die vorab geschriebene Schwelle 'M1 verschiebt um > 0,1' war eine Fehlerwartung aus dem Kopf;
# exakt gilt M1 - base = (1/2) g_{-1}(0,18) = 0,02170 (nachgerechnet im Gate selbst).
g1 = g(-1, tq(18, 100))
exp_shift = g1 / 2
check("G6_mutanten_verschieben", (abs((m1 - base) - exp_shift) < arb("1e-15")) and (m2 - base) > 0.0,
      f"base {base.str(8)} M1 {m1.str(8)} (erwartet Verschiebung {exp_shift.str(8)}) M2 {m2.str(8)}")

# ---------------------------------------------------------------- G7 Zertifikats-Maschine: Schwellen-Eichung (vorab bekannt: min F = 0,0018456 bei t=0,18)
a, b = arb(fmpq(18, 100)), arb(1)
r15 = certify_adaptive(19, arb(23), a, b, arb("0.0015"))
r18 = certify_adaptive(19, arb(23), a, b, arb("0.0018"))
r19 = certify_adaptive(19, arb(23), a, b, arb("0.0019"), min_w=arb(2) ** -20)
out(f"  G7 Schwelle 0,0015: ok={r15['ok']} Kaesten {r15['boxes']}; 0,0018: ok={r18['ok']} Kaesten {r18['boxes']}; 0,0019: ok={r19['ok']} Fehlkaesten {len(r19['fail_boxes'])}")
check("G7_schwelle_0,0015_besteht", r15['ok'])
check("G7_schwelle_0,0018_besteht_und_teilt", r18['ok'] and r18['splits'] > r15['splits'])
check("G7_schwelle_0,0019_FAELLT", (not r19['ok']) and len(r19['fail_boxes']) >= 1)
# Fehlkaesten liegen am linken Rand
if r19['fail_boxes']:
    lo, hi, _ = r19['fail_boxes'][0]
    check("G7_fehlkasten_am_linken_rand", float(hi.mid()) < 0.19, f"Fehlkasten [{float(lo.mid()):.6f}, {float(hi.mid()):.6f}]")
# Kontrolle (R=15,p=19): muss auf [0,18; 0,1967] sicher NEGATIV sein, d.h. Schwelle 0 faellt
rc = certify_adaptive(15, arb(19), arb(fmpq(18, 100)), arb(fmpq(1967, 10000)), arb(0), min_w=arb(2) ** -20)
check("G7_kontrolle_R15_p19_faellt", not rc['ok'])
# Taylor-Maschine: gleiche Eichung
tt15 = certify_taylor(19, 23, a, arb("0.92"), arb("0.0015"))
tt19 = certify_taylor(19, 23, a, arb("0.92"), arb("0.0019"), min_w=arb(2) ** -20)
check("G7_taylor_0,0015_besteht", tt15['ok'], f"Kaesten {tt15['boxes']}")
check("G7_taylor_0,0019_FAELLT", not tt19['ok'])

# ---------------------------------------------------------------- G8 sinc-Reihe: Einschluss und Weite am Rand s -> 0 (Kasten enthaelt 0)
okk = True
for s_hi in ["0.001", "0.05", "0.3"]:
    sb = arb(0).union(arb(s_hi))
    enc = sinc2_series(sb)
    for k in range(0, 21):
        sv = mp.mpf(s_hi) * k / 20
        mp.mp.dps = 40
        tv = (mp.sinc(mp.pi * sv)) ** 2
        lo = mp.mpf(enc.lower().mid().str(40, radius=False)); hi = mp.mpf(enc.upper().mid().str(40, radius=False))
        okk &= (lo <= tv <= hi)
    out(f"  G8 s in [0,{s_hi}]: sinc^2 in {enc.str(12)}")
check("G8_sinc_reihe_einschluss", okk)
check("G8_sinc_weite", (arb(0).union(arb("0.001")))*0 == 0 and sinc2_series(arb(0).union(arb("0.001"))).rad() < arb("1e-5"))

# ---------------------------------------------------------------- G9 Praezisionsunabhaengigkeit des Urteils
ctx.prec = 100
r_lo = certify_adaptive(19, arb(23), arb(fmpq(18, 100)), arb(1), arb("0.0015"))
ctx.prec = 300
r_hi = certify_adaptive(19, arb(23), arb(fmpq(18, 100)), arb(1), arb("0.0015"))
ctx.prec = 200
check("G9_urteil_praezisionsunabhaengig", r_lo['ok'] and r_hi['ok'], f"100 bit: {r_lo['boxes']} Kaesten, 300 bit: {r_hi['boxes']}")

# ---------------------------------------------------------------- G10 Restglied T_R >= exaktes Hurwitz-Restglied (Stichprobe)
okt = True; mind = 1e9
for k in range(1, 40):
    t = arb(fmpq(k, 40))
    st2 = t.sin_pi() ** 2 / (arb.pi() ** 2)
    Z = st2 * (arb(2).zeta(19 + 1 + t) + arb(2).zeta(19 + 1 - t))
    d = TR(19, t) - Z
    okt &= (d >= 0)
    mind = min(mind, float(d.mid()) / max(float(TR(19, t).mid()), 1e-300))
check("G10_restglied_T19_ueber_hurwitz", okt, f"39 Punkte, kleinster relativer Abstand {mind:.2e}")

allok = all(res.values())
out("")
out(f"GATE E7926: {'GRUEN' if allok else 'ROT'}  ({sum(res.values())}/{len(res)} Pruefungen)")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
sys.exit(0 if allok else 1)
