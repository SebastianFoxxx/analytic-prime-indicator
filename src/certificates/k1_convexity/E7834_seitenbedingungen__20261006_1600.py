"""E7834 (T36-F-1198, R1/R1b) SEITENBEDINGUNGEN der Zertifikate E7828/E7833 (C7: Voraussetzungen der Rechnung am Objekt messen).

VORAB-REGELN (vor dem Lauf):
 S1 g_0''(t) < 0 auf ganz [0, 9/50] (Ball-Obergrenze ueber das ganze Intervall; noetig fuer 'W(0) <= p => W g_0'' >= p g_0'''').
 S2 Randform: V(t) > 0 (untere Schranke) auf allen t-Boxen der Breite 0,001 ueber [0, 0,18] fuer jedes p in {5,7,11,13,17,19} und fuer das u-Intervall [0, 1/24] (p-uniform): noetig fuer 'V n(s) >= V_low n_low'.
 S3 A1 >= 0 (untere Schranke) auf denselben Boxen (Koeffizienten c_b >= 0, g_rho'' > 0): noetig fuer 'sigma(s) A1 >= sigma_lo A1_lo'.
 S4 Ueberdeckung: die t-Boxen der Zertifikate E7828 (6 Primzahlen) und E7833 (2 Klassen) ueberdecken [0, 9/50] lueckenlos (sortiert, Anschluss exakt, letzte Obergrenze >= 9/50, erste = 0).
 S5 Kontrolle der Eingaben: s-Gitter ueberdeckt [0, 40] lueckenlos und der Rest [40, oo) wird separat geschuetzt (sigma(40) >= 1 - 5,0e-18, |m| <= 40^2 e^-40).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import arb
import p1198lib as L
import p1198uni as U

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


res = {}
T = arb(9) / 50
g = L.g0pp(arb(T.mid() / 2, T.mid() / 2 * 1.0000001))
log("S1: g0'' auf [0, 9/50]: Ball", g, " Obergrenze", float(g.upper()))
res["S1"] = bool(g.upper() < 0)
# S2/S3
bad2 = 0
bad3 = 0
nbox = 0
w = 0.001
boxes_t = [(k * w, (k + 1) * w) for k in range(180)] + [(0.18, 0.18000000000000002)]
for p in (5, 7, 11, 13, 17, 19):
    sigma0 = L.sig(2 * arb(3.25) * arb(p).log())
    data = L.divisor_data(p, 600)
    for t0, t1 in boxes_t:
        t = arb((t0 + t1) / 2, (t1 - t0) / 2 * 1.0000001)
        A0, A1 = L.A_coeffs(p, t, sigma0, data)
        Uu, Vv = L.edge_UV(p, t)
        nbox += 1
        if not (Vv.lower() > 0):
            bad2 += 1
        if not (A1.lower() >= 0):
            bad3 += 1
for r, plo in ((1, 31), (5, 23)):
    cf = U.class_coeffs(r, p_lo=plo)
    sigma0 = L.sig(2 * arb(3.25) * arb(plo).log())
    ub = [arb((a + b) / 2, (b - a) / 2 * 1.0000001) for a, b in zip((0.0, 1 / 100), (1 / 100, 1 / (plo + 1)))]
    for t0, t1 in boxes_t:
        t = arb((t0 + t1) / 2, (t1 - t0) / 2 * 1.0000001)
        A0, A1 = U.A_coeffs_uniform(t, sigma0, cf)
        for uu in ub:
            Uq, Vq = U.edge_UV_u(t, uu)
            nbox += 1
            if not (Vq.lower() > 0):
                bad2 += 1
        if not (A1.lower() >= 0):
            bad3 += 1
log(f"S2: V > 0: Verletzungen {bad2} / {nbox} Boxen ; S3: A1 >= 0: Verletzungen {bad3}")
res["S2"] = bad2 == 0
res["S3"] = bad3 == 0
# S4
grid = L.s_grid()
sb = L.s_boxes_lower(grid)
ok4 = True
for name, rr in [(f"p={p}", L.certify(p, 3.25, R=600, first_split=64, grid=grid, sb=sb, tol_width=1e-6)) for p in (5, 7, 11, 13, 17, 19)] + \
        [(f"Klasse {r}", U.certify_uniform(r, 3.25, p_lo=plo, grid=grid, sb=sb, first_split=64, tol_width=1e-6, u_split=(0.0, 1 / 100, 1 / (plo + 1)))) for r, plo in ((1, 31), (5, 23))]:
    bx = sorted(rr["boxes"], key=lambda d: d[0])
    cont = bx[0][0] == 0.0 and all(bx[k][1] == bx[k + 1][0] for k in range(len(bx) - 1)) and bx[-1][1] >= 0.18
    log(f"S4: {name}: ok={rr['ok']} Boxen {len(bx)}  lueckenlos={cont}  [{bx[0][0]}, {bx[-1][1]}]")
    ok4 = ok4 and cont and rr["ok"]
res["S4"] = ok4
# S5
cover = all(abs(grid[k][1] - grid[k + 1][0]) < 1e-12 for k in range(len(grid) - 1)) and grid[0][0] == 0.0 and abs(grid[-1][1] - 40.0) < 1e-12
tail_ok = bool((1 - L.sig(arb(40))) < arb(5.0e-18)) and bool(-(arb(40) ** 2 * (-arb(40)).exp()) > -1e-14)
log(f"S5: s-Gitter lueckenlos {cover}; Rest-Schutz {tail_ok}")
res["S5"] = cover and tail_ok
log("Ergebnis:", res)
with open(os.path.join(HERE, "logs", "E7834_seitenbedingungen__20261006_1600.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
