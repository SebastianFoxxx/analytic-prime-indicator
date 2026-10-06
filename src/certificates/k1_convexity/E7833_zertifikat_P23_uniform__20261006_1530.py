"""E7833 (T36-F-1198, R1b) MESSLAUF: p-uniformes Intervallzertifikat K1' fuer ALLE Primzahlen p >= 23 (kappa >= C (p+1) ln p, t in [0, 9/50]).

VORAB-REGELN / ERWARTUNGEN (vor dem Lauf; Gate E7832 gruen, Lauf 2):
 X1 C = 3,25: certify_uniform ok = True fuer Klasse 1 (p >= 31) UND Klasse 5 (p >= 23), Zielmarge P''/p >= 0,5.
    Erwartung aus einem ENTWICKLUNGSLAUF VOR dem Gate (keine Messung): ca. 0,80 (Klasse 1), 0,73 (Klasse 5) -> Zielmarge 0,5 sollte klar fallen.
 X2 Kleinstes zertifiziertes C je Klasse auf dem Raster {1,0; 1,25; 1,5; 1,75; 2,0; 2,5; 3,25} (Marge > 0).  Erwartung: <= 1,5 (Entwicklungslauf: C = 1,5 ok, 1,0 nicht geprueft).
 X3 Das Ergebnis gilt fuer ALLE p >= p_lo der Klasse, weil (i) die Klassenkoeffizienten das Minimum ueber alle ganzen q der Klasse in [p_lo, 300] und den Limes sind (W1: exakt gegen alle Primzahlen <= 5000 geprueft, 0 Verletzungen),
    (ii) u = 1/(p+1) als Ball in [0, 1/(p_lo+1)] laeuft, (iii) E(p) faellt (W4), (iv) kappa-Einschraenkung s >= s_min(t) mit p_lo.
 Gueltigkeit: nur wenn Gate E7832 gruen und einfo['ratio_ok'].
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1198lib as L
import p1198uni as U

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


grid = L.s_grid()
sb = L.s_boxes_lower(grid)
PL = {1: 31, 5: 23}
res = {}
log("E7833 Zertifikat P23 uniform", time.strftime("%Y-%m-%d %H:%M"))
for r in (1, 5):
    plo = PL[r]
    cf = U.class_coeffs(r, p_lo=plo)
    ush = (0.0, 1 / 100, 1 / (plo + 1))
    got = None
    for mg in (0.5, 0.25, 0.1, 0.0):
        rr = U.certify_uniform(r, 3.25, p_lo=plo, grid=grid, sb=sb, first_split=64, margin=mg, tol_width=1e-6, coeffs=cf, u_split=ush)
        if rr["ok"]:
            got = mg
            break
    mo = rr["min_over_p"]
    lo = float(mo.lower()) if mo is not None else None
    log(f"Klasse {r} (p >= {plo}) C=3,25: ok={rr['ok']} Zielmarge={got}  P''/p >= {lo}  bei t in [{rr['argmin_box'][0]:.5f}, {rr['argmin_box'][1]:.5f}]  E={float(rr['E'].upper()):.3e}  ratio_ok={rr['einfo']['ratio_ok']}  t-Boxen={rr['n_boxes']}")
    res[f"klasse{r}"] = dict(ok=rr["ok"], margin_target=got, min_over_p=lo, E=float(rr["E"].upper()), n_boxes=rr["n_boxes"], ratio_ok=rr["einfo"]["ratio_ok"], p_lo=plo)
    cmin = None
    for C in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.25):
        r2 = U.certify_uniform(r, C, p_lo=plo, grid=grid, sb=sb, first_split=32, margin=0.0, tol_width=2e-3, coeffs=cf, u_split=ush)
        if r2["ok"]:
            cmin = C
            break
    log(f"Klasse {r}: kleinstes zertifiziertes C auf dem Raster = {cmin}")
    res[f"klasse{r}"]["C_cert"] = cmin
with open(os.path.join(HERE, "out", "E7833_ergebnis.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
with open(os.path.join(HERE, "logs", "E7833_zertifikat_P23__20261006_1530.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
