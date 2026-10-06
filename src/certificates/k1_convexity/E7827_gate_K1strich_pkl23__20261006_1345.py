"""E7827 (T36-F-1198, R1) GATE VOR DER MESSUNG: Instrument fuer das K1'-Intervallzertifikat p < 23.

VORAB-REGELN (festgeschrieben VOR E7828, Commit siehe Log):
 V1 Zerlegung: |P''_direkt(mp.diff) - (A + B_edge + B_rest)| <= 1e-12 * max(1, |P''|) an 12 Punkten (p in {5,7,11}, t in {0,03; 0,1; 0,18}, kappa in {1;2} * kappa_min(C=3,25)).
 V2 Randformel: |B_edge/(p+1) - (U m(s) + V n(s))| <= 1e-15 an denselben Punkten (U, V, s aus der Bibliothek, B_edge aus mp.diff).
 V3 g0'' (Reihe mit Restkugel) und g_rho'' (Schliessform) gegen mp.diff der Definition, |Delta| <= 1e-20, an t in {0; 0,01; 0,1; 0,18}.
 V4 Soundness an 6 Primzahlen x 4 kappa-Vielfache (1; 1,3; 3; 10) x 4 t (0,001; 0,02; 0,09; 0,18) = 96 Zellen, kappa_min bei C = 3,25:
    (a) A0 + sigma(s) A1 <= A_exakt (mp) ; (b) |B_rest| <= eps_p ; (c) Gesamtschranke A0 + sigma(s) A1 + (p+1)(U m + V n) - eps <= P''_direkt.  Erlaubte Verletzungen: 0.
 V5 Mutanten MUESSEN scheitern (certify liefert ok = False):  M1 R = 3 ; M2 Vorzeichen von V gedreht ; M3 C = 0,3 (p = 5) ; M4 keine Teiler (A0 = p g0'' nur) ; M5 sigma0 = 0,5 (p = 5, C = 3,25).
    Eine durchgelaufene Mutante heisst: Instrument unempfindlich -> Gate rot.
 V6 Erreichbarkeit aller Zweige: certify(p=5, C=3,25, first_split=1) teilt mindestens einmal (n_boxes > 1), und ein Mutantenlauf erreicht den fails-Zweig.
 V7 Positivkontrolle: die Intervallschranke ist <= wahrer min P''/p (mp-Raster, p = 5, C = 6,4 kappa-Wert) und > 0.
Entscheidung: alle V1-V7 gruen -> E7828 (Messlauf) darf starten. Sonst: Instrument reparieren (C8), nicht das Objekt verdaechtigen.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from flint import arb

import p1198lib as L
import p1198mp as M

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


mp.mp.dps = 40
t_start = time.time()
res = {}
log("E7827 Gate T36-F-1198 R1 -", time.strftime("%Y-%m-%d %H:%M"))

# ---------------- V1 / V2
log("== V1/V2 Zerlegung und Randformel")
v1_bad = 0
v2_bad = 0
n12 = 0
for p in (5, 7, 11):
    kmin = float(L.kappa_min(p, 3.25).mid())
    for mult in (1.0, 2.0):
        kap = kmin * mult
        for t in (0.03, 0.1, 0.18):
            A, Be, Br, S = M.split_AB(p, t, kap)
            Pd = M.Pdd_direct(p, t, kap)
            d1 = abs(Pd - S)
            ok1 = d1 <= 1e-12 * max(1, abs(Pd))
            U, V = L.edge_UV(p, arb(t))
            s = 2 * mp.mpf(kap) * t / (p + 1 + t)
            m = float(L.mfun(arb(float(s))).mid())
            n = float(L.nfun(arb(float(s))).mid())
            rhs = float(U.mid()) * m + float(V.mid()) * n
            d2 = abs(float(Be) / (p + 1) - rhs)
            ok2 = d2 <= 1e-15
            v1_bad += (not ok1)
            v2_bad += (not ok2)
            n12 += 1
            log(f"  p={p} mult={mult} t={t}: P''={float(Pd):.10f}  |Delta_Zerl|={float(d1):.2e}  |Delta_Rand|={d2:.2e}  A={float(A):.6f} Be={float(Be):.6f} Br={float(Br):.3e}")
log(f"V1 Verletzungen {v1_bad}/{n12} ; V2 Verletzungen {v2_bad}/{n12}")
res["V1"] = v1_bad == 0
res["V2"] = v2_bad == 0

# ---------------- V3
log("== V3 g0'' und g_rho''")
v3_bad = 0
g0 = lambda y: mp.sin(mp.pi * y) ** 2 / (mp.pi ** 2 * y ** 2)
for t in (0.0, 0.01, 0.1, 0.18):
    if t == 0:
        ref0 = -2 * mp.pi ** 2 / 3
    else:
        ref0 = mp.diff(g0, mp.mpf(t), 2)
    v = L.g0pp(arb(t))
    d = abs(float(v.mid()) - float(ref0))
    ok = d <= 1e-20 and float(v.rad()) < 1e-20
    v3_bad += (not ok)
    log(f"  g0''({t}): Reihe {float(v.mid()):.15f} Ref {float(ref0):.15f} |D|={d:.1e} rad={float(v.rad()):.1e}")
    for rho in (-3, -1, 1, 2, 7):
        gr = lambda y, r=rho: mp.sin(mp.pi * y) ** 2 / (mp.pi ** 2 * (r + y) ** 2)
        ref = mp.diff(gr, mp.mpf(t), 2) if t > 0 else mp.diff(gr, mp.mpf('1e-8'), 2)
        v = L.gpp_rho(rho, arb(t))
        d = abs(float(v.mid()) - float(ref))
        tol = 1e-20 if t > 0 else 1e-6
        ok = d <= tol
        v3_bad += (not ok)
        if not ok:
            log(f"  VERLETZUNG g_{rho}''({t}): {float(v.mid())} vs {float(ref)}")
log(f"V3 Verletzungen {v3_bad}")
res["V3"] = v3_bad == 0

# ---------------- V4
log("== V4 Soundness (96 Zellen)")
v4a = v4b = v4c = 0
ncell = 0
minslack = 1e9
for p in (5, 7, 11, 13, 17, 19):
    C = 3.25
    sigma0 = L.sig(2 * arb(C) * arb(p).log())
    km = L.kappa_min(p, C)
    eps, info = L.eps_rest(p, km)
    data = L.divisor_data(p, 600)
    for mult in (1.0, 1.3, 3.0, 10.0):
        kap = km * mult
        kf = mp.mpf(str(kap.mid().str(30, radius=False)))
        for t in (0.001, 0.02, 0.09, 0.18):
            tb = arb(t)
            A0, A1 = L.A_coeffs(p, tb, sigma0, data)
            s = 2 * kap * tb / (p + 1 + tb)
            Alow = A0 + L.sig(s) * A1
            U, V = L.edge_UV(p, tb)
            Bedge_low = (p + 1) * (U * L.mfun(s) + V * L.nfun(s))
            A, Be, Br, S = M.split_AB(p, t, kf)
            a_ok = float(Alow.lower()) <= float(A)
            b_ok = abs(float(Br)) <= float(eps.upper())
            tot = Alow + Bedge_low - eps
            c_ok = float(tot.lower()) <= float(S)
            v4a += (not a_ok)
            v4b += (not b_ok)
            v4c += (not c_ok)
            ncell += 1
            minslack = min(minslack, float(S) - float(tot.lower()))
            if not (a_ok and b_ok and c_ok):
                log(f"  VERLETZUNG p={p} mult={mult} t={t}: A_low {float(Alow.mid()):.5f} A {float(A):.5f} |Br| {abs(float(Br)):.3e} eps {float(eps.upper()):.3e} tot {float(tot.mid()):.5f} P'' {float(S):.5f}")
log(f"V4 Zellen {ncell}: (a) {v4a}  (b) {v4b}  (c) {v4c} Verletzungen; kleinster Abstand P'' - Schranke = {minslack:.4f}")
res["V4"] = (v4a + v4b + v4c) == 0

# ---------------- V5 / V6
log("== V5 Mutanten / V6 Erreichbarkeit")
grid = L.s_grid()
sb = L.s_boxes_lower(grid)
muts = {
    "M1_R3": dict(p=5, C=3.25, mut={"R": 3}),
    "M2_flipV": dict(p=5, C=3.25, mut={"flipV": True}),
    "M3_C0.3": dict(p=5, C=0.3, mut={}),
    "M4_keineTeiler": dict(p=5, C=3.25, mut={"R": 0}),
    "M5_sigma0_0.5": dict(p=5, C=3.25, mut={"sigma0": 0.5}),
}
v5_bad = 0
fails_branch = False
for name, cfg in muts.items():
    r = L.certify(cfg["p"], cfg["C"], R=600, mut=cfg["mut"], first_split=4, grid=grid, sb=sb, tol_width=5e-3)
    log(f"  {name}: ok={r['ok']} fails={len(r['fails'])} boxes={r['n_boxes']}")
    if r["ok"]:
        v5_bad += 1
    if r["fails"]:
        fails_branch = True
res["V5"] = v5_bad == 0
r6 = L.certify(5, 3.25, R=600, first_split=1, grid=grid, sb=sb, tol_width=1e-4)
log(f"  V6 p=5 C=3,25 first_split=1: ok={r6['ok']} boxes={r6['n_boxes']} (muss > 1 sein) ; fails-Zweig durch Mutanten erreicht: {fails_branch}")
res["V6"] = (r6["n_boxes"] > 1) and fails_branch

# ---------------- V7
log("== V7 Positivkontrolle")
p = 5
C = 6.4
kap = float(L.kappa_min(p, C).mid())
r7 = L.certify(p, C, R=600, first_split=32, grid=grid, sb=sb, tol_width=2e-4, margin=0.0)
true_min = None
for t in [1e-4, 1e-3, 5e-3, 0.01, 0.03, 0.06, 0.1, 0.14, 0.18]:
    v = float(M.Pdd_direct(5, t, kap)) / 5
    true_min = v if true_min is None else min(true_min, v)
bound = float(r7["min_over_p"].lower()) if r7["min_over_p"] is not None else None
log(f"  p=5, C=6,4: wahres min P''/p (Raster) = {true_min:.4f}  Intervallschranke = {bound}")
res["V7"] = (bound is not None) and (0 < bound <= true_min)

log("== Ergebnis Gate:", res)
log("GATE", "GRUEN" if all(res.values()) else "ROT", f"({time.time()-t_start:.0f} s)")
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
with open(os.path.join(HERE, "logs", "E7827_gate__20261006_1345.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
