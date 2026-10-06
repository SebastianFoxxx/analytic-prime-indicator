"""E8041 (T36-F-1260) MESSLAUF: Konvexitaets-Zertifikat p >= 23 pro Restklasse mit den EXAKTEN Klassenkoeffizienten (Lemma K),
d. h. ohne das Stellvertreterelement q = 10^9+7 des Altskripts E7833. Gleiche Boxen/Parameter wie E7833.

VORAB-REGELN:
 Z1  C = 3,25: certify_uniform ok = True fuer Klasse 1 (p >= 31) UND Klasse 5 (p >= 23) bei Zielmarge 0,5; P''/p-Untergrenze >= 0,908 (Klasse 1)
     bzw. >= 0,821 (Klasse 5), also dieselben im Paper genannten, abgerundeten Konstanten (Abweichung zum Altlauf < 1e-6).
 Z2  Die Abweichung der Untergrenzen gegenueber E7833 (0,9084107096323574 / 0,821911663062404) betraegt hoechstens 1e-6.
 Z3  Mutant: c_a * 1,01 (nur im Betrag erhoeht) darf das Zertifikat NICHT veraendern, wenn man ihn als Untergrenze benutzt, aber der Primzahltest
     E8040 F faellt; hier: Mutant c_a * 0 -> Untergrenze deutlich kleiner (Detektor reagiert auf die Koeffizienten).
Soundness von Arb angenommen.
"""
import os, sys, time, json
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "T36_F1198"))
import l1260lib as L
import p1198lib as PL_
import p1198uni as U

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)

t0 = time.time()
log("E8041 Zertifikat Klassen exakt", time.strftime("%Y-%m-%d %H:%M"))
grid = PL_.s_grid(); sb = PL_.s_boxes_lower(grid)
PLO = {1: 31, 5: 23}
old_ref = {1: 0.9084107096323574, 5: 0.821911663062404}
res = {}
ok_all = True
for r in (1, 5):
    plo = PLO[r]
    cf = L.class_coeffs_exact(r, plo)
    ush = (0.0, 1 / 100, 1 / (plo + 1))
    got = None
    for mg in (0.5, 0.25, 0.1, 0.0):
        rr = U.certify_uniform(r, 3.25, p_lo=plo, grid=grid, sb=sb, first_split=64, margin=mg, tol_width=1e-6, coeffs=cf, u_split=ush)
        if rr["ok"]:
            got = mg; break
    lo = float(rr["min_over_p"].lower())
    d = abs(lo - old_ref[r])
    log(f"Klasse {r} (p >= {plo}) C=3,25 exakte Koeffizienten: ok={rr['ok']} Zielmarge={got}  P''/p >= {lo!r}  Abweichung zu E7833: {d:.2e}  E={float(rr['E'].upper()):.4e} t-Boxen={rr['n_boxes']} argmin t in [{rr['argmin_box'][0]:.5f},{rr['argmin_box'][1]:.5f}]")
    z1 = rr["ok"] and got is not None and got >= 0.5 and lo > (0.908 if r == 1 else 0.821)
    z2 = d <= 1e-6
    ok_all = ok_all and z1 and z2
    res[f"klasse{r}"] = dict(ok=rr["ok"], margin=got, lower=lo, diff_E7833=d, E=float(rr["E"].upper()), boxes=rr["n_boxes"])
    # Z3 Mutant: c_a = 0 (alle ausser rho=0 Teil)
    cf0 = {k: (Fraction(0), v[1]) for k, v in cf.items()}
    rm = U.certify_uniform(r, 3.25, p_lo=plo, grid=grid, sb=sb, first_split=64, margin=0.0, tol_width=1e-4, coeffs=cf0, u_split=ush)
    lm = float(rm["min_over_p"].lower()) if rm["min_over_p"] is not None else None
    log(f"   Z3 Mutant c_a = 0: ok={rm['ok']} untere Schranke {lm} (deutlich kleiner als {lo:.3f} oder nicht zertifizierbar erwartet)")
    ok_all = ok_all and ((not rm["ok"]) or (lm is not None and lm < lo - 0.1))
log("ERGEBNIS:", "GRUEN" if ok_all else "ROT", f"({time.time()-t0:.0f} s)")
json.dump(res, open(os.path.join(HERE, "out", "E8041_ergebnis.json"), "w"), indent=1)
open(os.path.join(HERE, "logs", "E8041_zertifikat_klassen_exakt__20261006_2230.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
