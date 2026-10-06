"""E7828 (T36-F-1198, R1) MESSLAUF: Intervallzertifikat K1' fuer p in {5,7,11,13,17,19}, alle kappa >= C (p+1) ln p, t in [0, 9/50].

VORAB-REGELN / ERWARTUNGEN (festgeschrieben VOR dem Lauf; Gate E7827 gruen, Commit 2cf055919):
 Z1 Fuer C = 3,25 liefert certify ok = True fuer ALLE sechs Primzahlen (Zielmarge: P'' >= 0,5 p, d. h. margin = 0,5).
    Erwartung aus einem ERKUNDUNGSLAUF VOR dem Gate (Entwicklungslauf, nicht Messung): knapp erfuellt bei p = 13 (0,508 p); faellt Z1 bei einer Primzahl, wird die Marge ehrlich gesenkt (0,25 / 0,1 / 0) und ausgewiesen, nicht verschwiegen.
 Z2 Sensitivitaet: R = 1200 statt 600 aendert die Marge nur nach oben (R ist nur ein Abschneiden positiver Terme).
 Z3 Kleinstes zertifiziertes C je Primzahl auf dem Raster {1,0; 1,25; 1,5; 1,75; 2,0; 2,5; 3,25} (Marge > 0): Erwartung p = 5: > 1,0 (der Rest-Cutoff-Schranke eps ist bei C=1 gross).
 Z4 Die Zertifikate gelten fuer ALLE kappa >= kappa_min (nicht nur ein Raster): das beruht auf (i) A_low waechst in s (also in kappa), (ii) B_edge exakt in (t, s), (iii) eps_rest faellt in kappa (mono_ok, ratio_ok werden ausgewiesen).
 Gueltigkeit des Ergebnisses: nur wenn Gate E7827 gruen UND mono_ok/ratio_ok True.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1198lib as L

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


grid = L.s_grid()
sb = L.s_boxes_lower(grid)
res = {}
log("E7828 Zertifikat K1' p<23 -", time.strftime("%Y-%m-%d %H:%M"), f" s-Boxen {len(sb)}")
for p in (5, 7, 11, 13, 17, 19):
    t0 = time.time()
    got = None
    for mg in (0.5, 0.25, 0.1, 0.0):
        r = L.certify(p, 3.25, R=600, first_split=64, margin=mg, grid=grid, sb=sb, tol_width=1e-6)
        if r["ok"]:
            got = mg
            break
    info = r["eps_info"]
    mo = r["min_over_p"]
    log(f"p={p:2d} C=3,25: ok={r['ok']} Marge-Ziel={got}  min P''/p >= {float(mo.lower()) if mo is not None else None:.5f}"
        f"  bei t in [{r['argmin_box'][0]:.5f}, {r['argmin_box'][1]:.5f}]  eps_rest={float(r['eps'].upper()):.4e}  kappa_min={float(r['kappa_min'].mid()):.4f}"
        f"  sigma0={float(r['sigma0'].mid()):.8f}  t-Boxen={r['n_boxes']}  mono_ok={info['mono_ok']} ratio_ok={info['ratio_ok']}  ({time.time()-t0:.1f} s)")
    res[p] = dict(ok=r["ok"], margin_target=got, min_over_p=float(mo.lower()) if mo is not None else None,
                  eps=float(r["eps"].upper()), n_boxes=r["n_boxes"], kappa_min=float(r["kappa_min"].mid()),
                  argmin=list(r["argmin_box"][:2]) if r["argmin_box"] else None, mono_ok=info["mono_ok"], ratio_ok=info["ratio_ok"])
# Z2
log("== Z2 R = 1200 (Sensitivitaet)")
for p in (5, 13):
    r = L.certify(p, 3.25, R=1200, first_split=64, margin=0.0, grid=grid, sb=sb, tol_width=1e-6)
    log(f"p={p} R=1200: ok={r['ok']} min P''/p >= {float(r['min_over_p'].lower()):.5f}  (R=600: {res[p]['min_over_p']:.5f})")
    res[p]["min_R1200"] = float(r["min_over_p"].lower())
# Z3
log("== Z3 kleinstes zertifiziertes C je p (Marge > 0, Raster)")
cmin = {}
for p in (5, 7, 11, 13, 17, 19):
    found = None
    for C in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.25):
        r = L.certify(p, C, R=600, first_split=16, margin=0.0, grid=grid, sb=sb, tol_width=2e-3)
        if r["ok"]:
            found = C
            break
    cmin[p] = found
    log(f"p={p}: kleinstes zertifiziertes C auf dem Raster = {found}")
res["C_cert"] = cmin
with open(os.path.join(HERE, "out", "E7828_ergebnis.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
with open(os.path.join(HERE, "logs", "E7828_zertifikat__20261006_1420.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
log("fertig")
