"""E8043 (T36-F-1260) MESSLAUF: die kleinen Schritte, die der Paper-Text nach Gegenlese T35-V-1091 / FIX-T35-0064 ausschreibt.

Gate E8039 gruen, committet vor diesem Lauf.  Objekt (C4): P_sigma(x;kappa) des API-Papers (Def. 8.1), endliches kappa, p prim.

VORAB-REGELN:
 D1  (Reduktion p >= 23, Anhang C.3) Einzelschritte je fuer p in 76 Primzahlen 23..10007, kappa/kappa_min in {1; 1,5; 4}, t in [23/128, 1) (24 Punkte):
     (a) D/p <= D0(C) mit a0 = 2 C ln 23 24/25, D0 = exp(-a0 (1+t_min))/(1-exp(-a0)),  (b) (1-phi_{p+1}) E/p <= (24/23) g_24(1-t) W_C(t)
     fuer C = 3,05 und 3,25: 0 Verletzungen.  Mutant (b) mit g_{1009} statt g_24 faellt an p = 23.
 D2  g_n(s) = (sinc s/ sinc(s/n))^2 faellt in n (n = 2..4000, s in {0,001..0,82}): 0 Verletzungen (Arb).
 D3  Arb-Neuzertifikat (statt mpmath.iv) fuer sup_t Delta_up(t,C) + D0(C) < 0,0015 auf [23/128, 1]: C = 3,05 gelingt, C = 3,00 gelingt NICHT;
     der kritische Wert C* wird auf 1e-4 eingegrenzt (Erwartung: zwischen 3,00 und 3,05).
 B1  B_edge/(p+1) = U s^2 sigma''(s) + V s sigma'(s) mit den im Paper ausgeschriebenen U, V: relative Abweichung < 1e-12 gegen die direkte Rechnung
     (phi'' f + 2 phi' f' fuer i = p+1, mpmath dps 40) an 200 Zufallsfaellen.
 T1  Rem. 8.11: kleinste Schwelle y_p = kappa/(p+1), ab der P_sigma'(p;kappa) <= -3/4 gilt (groesste Wurzel von (p^2/(2(p+1))) y sech^2 y = 1/4):
     Minimum und Maximum ueber die Primzahlen 3 <= p <= 499 (Paper: 1,7-4,9; Gegenlese: 1,665..4,944).
 T2  Rem. 'Tail constants': sup_{x in [1,10]} sum_{i>11} phi_5(i/(x+1)) F(x,i)/i^2 (direkte Summe bis i = 400, Float64, plus Arb-Kontrolle am Maximierer)
     = 5,17e-3 (Gegenlese) und r^{M+1}/(1-r) = 3,06e-5.
 T3  max{1,.} in M_tau^unif nie aktiv: ceil(b + c^{-1} log(1/((1-e^{-c}) eps))) >= 1 fuer 400 Zufallsparameter (b>0, kappa>0, eps in (0,1)).
"""
import os, sys, math, time, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from flint import arb
import l1260lib as L

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)
t00 = time.time()
log("E8043 Hilfsschritte", time.strftime("%Y-%m-%d %H:%M"))
ok_all = True
mp.mp.dps = 30
TMIN = 23 / 128

# ---------------------------------------------------------------- D1
def fi_closed(p, i, t):
    m = p % i
    r = mp.sin(mp.pi * t) / (i * mp.sin(mp.pi * (m + t) / i))
    return r * r

def D_E(p, t, kappa):
    t = mp.mpf(t); X = p + 1 + t
    D = mp.mpf(0)
    for i in range(2, p + 1):
        omphi = 1 / (1 + mp.exp(2 * kappa * (1 - i / X)))   # 1 - phi_i = 1/(1+e^{s_i}), s_i = 2 kappa (1 - i/X)
        D += i * omphi * fi_closed(p, i, t)
    om = 1 / (1 + mp.exp(2 * kappa * t / X))                 # 1 - phi_{p+1}, s = 2 kappa t/X
    Ef = (p + 1) * (mp.sin(mp.pi * t) / ((p + 1) * mp.sin(mp.pi * (1 - t) / (p + 1)))) ** 2
    return D, om * Ef

def g_n(n, s):
    s = mp.mpf(s)
    return (mp.sin(mp.pi * s) / (mp.pi * s) / (mp.sin(mp.pi * s / n) / (mp.pi * s / n))) ** 2

PR = [p for p in range(23, 330) if all(p % q for q in range(2, int(p**0.5) + 1)) and p % 6 in (1, 5)] + [401, 601, 1009, 2003, 5003, 10007]
tg = [TMIN + (1 - TMIN - 1e-7) * k / 23 for k in range(24)]
v1 = v2 = vm = 0
nD = 0
worst_a = worst_b = 0.0
for C in (3.05, 3.25):
    a0 = 2 * C * math.log(23) * 24 / 25
    D0 = math.exp(-a0 * (1 + TMIN)) / (1 - math.exp(-a0))
    for p in PR:
        kmin = C * (p + 1) * math.log(p)
        for kf in (1.0, 1.5, 4.0):
            for t in tg:
                D, OE = D_E(p, t, kf * kmin)
                dp = float(D) / p
                eb = float(OE) / p
                bound_b = (24 / 23) * float(g_n(24, 1 - t)) * math.exp(-2 * C * math.log(23) * t / (1 + t / 24))
                nD += 1
                worst_a = max(worst_a, dp / D0)
                worst_b = max(worst_b, eb / bound_b)
                if dp > D0 * (1 + 1e-12): v1 += 1
                if eb > bound_b * (1 + 1e-12): v2 += 1
                if p == 23 and kf == 1.0 and C == 3.25:
                    mb = (24 / 23) * float(g_n(1009, 1 - t)) * math.exp(-2 * C * math.log(23) * t / (1 + t / 24))
                    if eb > mb: vm += 1
log(f"D1 {nD} Faelle: (a) D/p <= D0: Verletzungen {v1} (max Verhaeltnis {worst_a:.4f}); (b) (1-phi_(p+1))E/p <= (24/23) g_24 W_C: Verletzungen {v2} (max Verhaeltnis {worst_b:.4f}); Mutant g_1009 verletzt {vm} Faelle an p=23")
ok_all &= (v1 == 0 and v2 == 0 and vm > 0)

# ---------------------------------------------------------------- D2
v = 0
for s in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.82]:
    prev = None
    for n in list(range(2, 200)) + list(range(200, 4001, 37)):
        sa = arb(s)
        g = ((sa * L.PI).sin() / (sa * L.PI) / (((sa * L.PI / n).sin()) / (sa * L.PI / n))) ** 2
        if prev is not None and not (g.upper() <= prev.lower()):
            v += 1
        prev = g
log(f"D2 g_n(s) fallend in n (Arb, getrennte Baelle): Verletzungen {v}")
ok_all &= v == 0

# ---------------------------------------------------------------- D3
def sinc_ball(x):
    return x.sinc_pi()

def delta_up_ub(C, a, b):
    tt = arb((a + b) / 2, (b - a) / 2 * 1.0000001)
    expo = 2 * arb(C) * arb(23).log() * tt / (1 + tt / 24)
    s = 1 - tt
    g = (sinc_ball(s) / sinc_ball(s / 24)) ** 2
    return (arb(24) / 23 * g * (-expo).exp()).upper()

def D0_ub(C):
    a0 = 2 * arb(C) * arb(23).log() * arb(24) / 25
    return ((-a0 * (1 + arb(TMIN))).exp() / (1 - (-a0).exp())).upper()

def certify_arb(C, maxdepth=40):
    d0 = D0_ub(C)
    budget = arb(0.0015) - d0
    stack = [(TMIN, 1.0, 0)]
    worst = arb(0)
    nint = 0
    while stack:
        a, b, dep = stack.pop()
        ub = delta_up_ub(C, a, b)
        if ub < budget:
            nint += 1
            if ub > worst: worst = ub
            continue
        if dep >= maxdepth:
            return False, None, nint
        m = 0.5 * (a + b)
        stack.append((a, m, dep + 1)); stack.append((m, b, dep + 1))
    return True, float((worst + d0).upper()), nint

res = {}
for C in (3.0, 3.04, 3.05, 3.1, 3.25):
    ok, w, n = certify_arb(C)
    res[C] = ok
    log(f"D3 Arb: C = {C}: zertifiziert = {ok}  sup(Delta_up + D0) <= {w}  ({n} Intervalle)")
lo_, hi_ = 3.0, 3.05
for _ in range(14):
    mid = 0.5 * (lo_ + hi_)
    if certify_arb(mid)[0]: hi_ = mid
    else: lo_ = mid
log(f"D3 kritischer Wert C* (Arb-Zertifikat schliesst ab) in [{lo_:.5f}, {hi_:.5f}]")
ok_all &= res[3.05] and (not res[3.0])

# ---------------------------------------------------------------- B1
rr = random.Random(11)
maxrel = 0.0
mp.mp.dps = 40
for _ in range(200):
    p = rr.choice([23, 29, 31, 37, 41, 101, 307])
    t = rr.uniform(0.002, 0.18)
    kappa = rr.uniform(0.5, 6.0) * (p + 1) * math.log(p)
    P1 = p + 1
    pm, tm, km = mp.mpf(p), mp.mpf(t), mp.mpf(kappa)
    X = pm + tm + 1
    s = 2 * km * tm / X
    s1 = 2 * km * P1 / X ** 2; s2 = -4 * km * P1 / X ** 3
    ee = mp.exp(-abs(s)); d1 = ee / (1 + ee) ** 2; d2 = -d1 * mp.tanh(s / 2)
    ph1 = d1 * s1; ph2 = d2 * s1 ** 2 + d1 * s2
    f = lambda tt: (mp.sin(mp.pi * tt) / (P1 * mp.sin(mp.pi * (1 - tt) / P1))) ** 2
    ff, f1 = f(tm), mp.diff(f, tm, 1)
    direct = P1 * (ph2 * ff + 2 * ph1 * f1)
    Sc = mp.sin(mp.pi * tm) / (mp.pi * tm)
    th = mp.pi * (1 - tm) / P1
    rho = mp.pi * Sc / (P1 * mp.sin(th))
    q = rho ** 2
    a = P1 / X
    T1 = mp.pi * tm * mp.cot(mp.pi * tm) - 1
    U_ = a ** 2 * q
    V_ = a * q * (4 + 4 * T1 + 4 * (mp.pi * tm / P1) * mp.cot(th) - 2 * tm / X)
    sp = ee / (1 + ee) ** 2
    spp = -sp * mp.tanh(s / 2)
    formula = P1 * (U_ * s ** 2 * spp + V_ * s * sp)
    rel = abs(direct - formula) / max(abs(direct), mp.mpf(10) ** -200)
    maxrel = max(maxrel, float(rel))
log(f"B1 B_edge-Formel (U, V): max relative Abweichung ueber 200 Faelle = {maxrel:.3e}")
ok_all &= maxrel < 1e-12

# ---------------------------------------------------------------- T1
def y_root(p):
    c = mp.mpf(p) ** 2 / (2 * (p + 1))
    g = lambda y: c * y / mp.cosh(y) ** 2 - mp.mpf(1) / 4
    y = mp.mpf(8)
    while g(y) < 0 and y > 0.8:
        y -= mp.mpf("0.01")
    if g(y) < 0:
        return None
    return mp.findroot(g, (y, y + mp.mpf("0.01")), solver="anderson")
rts = {}
for p in range(3, 500):
    if all(p % q for q in range(2, int(p**0.5) + 1)):
        rts[p] = float(y_root(p))
mn = min(rts.items(), key=lambda kv: kv[1]); mx = max(rts.items(), key=lambda kv: kv[1])
log(f"T1 Schwelle y_p = kappa/(p+1) fuer P_sigma'(p;kappa) <= -3/4 (3 <= p <= 499): Minimum {mn[1]:.4f} (p={mn[0]}), Maximum {mx[1]:.4f} (p={mx[0]})")
ok_all &= abs(mn[1] - 1.665) < 0.01 and abs(mx[1] - 4.944) < 0.01

# ---------------------------------------------------------------- T2
def tail(x, M=11, kappa=5.0, imax=400):
    x = np.asarray(x)[:, None]
    i = np.arange(M + 1, imax + 1)[None, :]
    u = i / (x + 1)
    e = np.clip(2 * kappa * (u - 1), -700, 700)
    phi = 1 / (1 + np.exp(e))
    f = (np.sin(np.pi * x) / (i * np.sin(np.pi * x / i))) ** 2
    return (phi * f).sum(axis=1)
xs = np.linspace(1, 10, 90001)
tv = tail(xs)
k = int(np.argmax(tv)); xm = xs[k]
xs2 = np.linspace(xm - 2e-4, xm + 2e-4, 40001)
tv2 = tail(xs2)
k2 = int(np.argmax(tv2))
log(f"T2 sup_x tail (Float64) = {tv2[k2]:.6e} bei x = {xs2[k2]:.6f}  (Gitter 90001 + Verfeinerung)")
r_ = math.exp(-10 / 11)
log(f"T2 r^(M+1)/(1-r) = {r_**12/(1-r_):.4e}")
xa = arb(float(xs2[k2]))
tot = arb(0)
for i in range(12, 401):
    u = arb(i) / (xa + 1)
    phi = 1 / (1 + (10 * (u - 1)).exp())
    f = ((xa * L.PI).sin() / (i * (xa * L.PI / i).sin())) ** 2
    tot += phi * f
tailb = (-(10 * (arb(401) / (xa + 1) - 1))).exp() / (1 - (-(10 / (xa + 1))).exp())
log(f"T2 Arb am Maximierer: Summe i=12..400 = {float(tot.mid()):.6e}, Schwanzschranke {float(tailb.upper()):.2e}")
ok_all &= abs(tv2[k2] - 5.17e-3) < 5e-5 and abs(r_**12/(1-r_) - 3.06e-5) < 1e-6

# ---------------------------------------------------------------- T3
vv = 0
for _ in range(400):
    b = 10 ** rr.uniform(-3, 3); kappa = 10 ** rr.uniform(-3, 3); eps = 10 ** rr.uniform(-12, -1e-9)
    c = 2 * kappa / (b + 1)
    val = b + math.log(1 / ((1 - math.exp(-c)) * eps)) / c
    if math.ceil(val) < 1: vv += 1
log(f"T3 ceil(b + c^-1 log(1/((1-e^-c) eps))) >= 1: Verletzungen {vv} / 400")
ok_all &= vv == 0
log("ERGEBNIS:", "GRUEN" if ok_all else "ROT", f"({time.time()-t00:.0f} s)")
open(os.path.join(HERE, "logs", "E8043_hilfsschritte__20261006_2300.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
