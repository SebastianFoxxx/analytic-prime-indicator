# -*- coding: utf-8 -*-
"""E7927 - K2 (T36-F-993-T1) in Kugelarithmetik (python-flint/Arb): Zertifikat F(t) = L_19(23,t) > 0,0015 auf [0,18; 1],
p-gleichmaessig ueber die Zerlegung L_19(p,t) = U(t) - V(t)/p. Gate: E7926 (Commit 956af77d4), Regeln GATE_E7926_E7933.md.
Zwei Mechaniken: (A) adaptive Bisektion, (B) Taylor-Modell 2. Ordnung (t <= 0,92) + adaptive s-Form (t >= 0,92).
"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2lib import *
from flint import arb, fmpq, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "E7927_zertifikat_K2__20261006_1900.txt")
lines = []
def out(s=""):
    print(s); lines.append(s)

t0 = time.time()
A, B = arb(fmpq(18, 100)), arb(1)
THR = arb("0.0015")
out("E7927 K2-Zertifikat in Arb, ctx.prec = %d Bit" % ctx.prec)
out("Objekt (C4): P_inf (kappa=inf), p ungerade Primzahl >= 23, t in [0,18; 1). Schwelle 0,0015 (Satz K2).")

# ---------- (A) adaptiv, p = 23
rA = certify_adaptive(19, arb(23), A, B, THR)
out(f"(A) adaptiv F>0,0015 auf [0,18;1]: ok={rA['ok']}, Kaesten {rA['boxes']}, Teilungen {rA['splits']}, kleinste Kasten-Untergrenze {rA['min_lower'].str(10)} (KEINE Marge, haengt an der Kastenbreite)")

# ---------- (B) Taylor-Modell [0,18; 0,92] + adaptiv [0,92; 1]
rB1 = certify_taylor(19, 23, A, arb("0.92"), THR)
rB2 = certify_adaptive(19, arb(23), arb("0.92"), B, THR)
out(f"(B) Taylor [0,18;0,92]: ok={rB1['ok']}, Kaesten {rB1['boxes']}, kleinste Untergrenze {rB1['min_lower'].str(10)}")
out(f"    adaptiv [0,92;1] (s-Form, sinc-Reihe): ok={rB2['ok']}, Kaesten {rB2['boxes']}, kleinste Untergrenze {rB2['min_lower'].str(10)}")

# ---------- (C) Punktwerte als Ball und rigoroser Rastermin (1999 Punkte, exakte rationale t)
pts = []
mn = None
for k in range(0, 2001):
    t = arb(fmpq(180000 + k * 410, 1000000))  # 0,18 .. 1,0 in 2000 Schritten
    v = L(19, arb(23), t)
    pts.append((float(t.mid()), float(v.mid())))
    lw = v.lower()
    if mn is None or lw < mn[1]:
        mn = (k, lw, t)
F18 = L(19, arb(23), arb(fmpq(18, 100)))
out(f"(C) F(0,18) = {F18.str(15)}  (Marge zu 0,0015: {(F18 - THR).str(8)}, relativ {((F18 - THR) / THR).str(5)})")
out(f"    Rastermin (2001 exakte rationale Punkte, Ball-Untergrenze): {mn[1].str(12)} bei t = {float(mn[2].mid()):.6f}")
tmin_idx = min(range(len(pts)), key=lambda i: pts[i][1])
out(f"    Float-Rastermin {pts[tmin_idx][1]:.8f} bei t = {pts[tmin_idx][0]:.6f} (Minimum am linken Rand: {tmin_idx == 0})")

# ---------- (D) p-gleichmaessig: L_19(p,t) = U(t) - V(t)/p
fU = lambda t: U_V(19, t)[0]
fVneg = lambda t: -U_V(19, t)[1]
# u0: grosszuegig abgerundet gegen den Rastermin von U
ugrid = []
vgrid = []
for k in range(0, 2001):
    t = arb(fmpq(180000 + k * 410, 1000000))
    U, V = U_V(19, t)
    ugrid.append(float(U.mid())); vgrid.append(float(V.mid()))
u_min, v_max = min(ugrid), max(vgrid)
out(f"(D) Raster: min U = {u_min:.8f} (bei t = {0.18 + 0.00041*ugrid.index(u_min):.5f}), max V = {v_max:.8f}")
u0 = arb(f"{u_min*0.995:.8f}")
v0 = arb(f"{v_max*1.005:.8f}")
rU = certify_adaptive(19, None, A, B, u0, fun=fU)
rV = certify_adaptive(19, None, A, B, -v0, fun=fVneg)
out(f"    U > u0 = {u0.str(8)} auf [0,18;1]: ok={rU['ok']} Kaesten {rU['boxes']};  V < v0 = {v0.str(8)}: ok={rV['ok']} Kaesten {rV['boxes']}")
pstar = None
if rU['ok'] and rV['ok']:
    pstar = v0 / (u0 - THR)
    out(f"    => fuer alle p > p* = v0/(u0-0,0015) = {pstar.str(8)} gilt L_19(p,t) >= u0 - v0/p > 0,0015 auf [0,18;1] (rigoros, ohne Monotonie)")
    out(f"    Grenzfall p -> inf: L_19(inf,t) = U(t) >= u0 = {u0.str(8)} = {(u0/THR).str(5)} x 0,0015")
# V > 0 (Monotonie ab p = 23: L(p,t) steigt in p genau wenn V > 0)
rV0 = certify_adaptive(19, None, A, B, arb(0), fun=lambda t: U_V(19, t)[1])
out(f"    V > 0 auf [0,18;1] (Monotonie: L_19(p,t) steigt in p): ok={rV0['ok']}, Kaesten {rV0['boxes']}")

# ---------- (E) Stichprobe p: direktes Zertifikat fuer ausgewaehlte p (nur Gegenprobe der Monotonie)
rows = []
for p in [23, 29, 31, 37, 101, 1009, 10007, 1000003]:
    r = certify_adaptive(19, arb(p), A, B, THR)
    Fp = L(19, arb(p), A)
    rows.append((p, r['ok'], r['boxes'], float(Fp.mid())))
    out(f"(E) p={p}: Zertifikat ok={r['ok']}, Kaesten {r['boxes']}, L_19(p;0,18) = {Fp.str(10)}")
okE = all(r[1] for r in rows)
incE = all(rows[i][3] < rows[i + 1][3] for i in range(len(rows) - 1))
out(f"    Monotonie in p am Rand t = 0,18 sichtbar: {incE}")

# ---------- Ergebnis
allok = rA['ok'] and rB1['ok'] and rB2['ok'] and rU['ok'] and rV['ok'] and rV0['ok'] and okE
out("")
out(f"ERGEBNIS E7927: {'K2 IN ARB ZERTIFIZIERT (beide Mechaniken, p-gleichmaessig)' if allok else 'NICHT ZERTIFIZIERT - Luecke beziffern'}   Laufzeit {time.time()-t0:.1f}s")
with open(LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(HERE, "out", "E7927_Fgrid.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["t", "F_mid", "U_mid", "V_mid"])
    for (t, f), u, v in zip(pts, ugrid, vgrid):
        w.writerow([f"{t:.8f}", f"{f:.12f}", f"{u:.12f}", f"{v:.12f}"])
json.dump(dict(A=rA['ok'], B1=rB1['ok'], B2=rB2['ok'], U=rU['ok'], V=rV['ok'], V0=rV0['ok'],
               u0=u0.str(10), v0=v0.str(10), pstar=(pstar.str(10) if pstar is not None else None),
               F018=F18.str(20), boxesA=rA['boxes'], boxesB1=rB1['boxes'], boxesB2=rB2['boxes']),
          open(os.path.join(HERE, "out", "E7927_result.json"), "w"))
sys.exit(0 if allok else 1)
