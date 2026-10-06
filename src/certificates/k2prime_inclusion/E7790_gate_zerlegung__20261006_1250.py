"""E7790  T36-F-1195 R2-Gate: Instrument- und Identitaetspruefung VOR den Messungen E7791-E7794.

Geprueft wird (eigener Code p1195lib gegen mpmath-Direktdefinition, dps 40, nicht gegen die Skripte von F1021/F1175):
 V1  Zerlegung P = Pinf - om*E - D + H  : max |Differenz| < 1e-9 (relativ zu p) an p in {5,7,23,101}, C in {1; 3,1; 6,4}, t in {0,18; 0,3; 0,6; 0,9; 0,999}
 V2  P(float) gegen P_mp(Definition): max |Differenz| < 1e-8 an denselben Zellen
 V3  Lam <= P (H >= 0) und H >= 0, D >= 0, om in (0, 1/2]  an denselben Zellen
 V4  Lam monoton in kappa: Lam(kappa=C1..) <= Lam(2 kappa) <= Lam(4 kappa) an denselben Zellen
 V5  P(p;kappa) geschlossene Form = P_mp(p, t=1e-20? -> Grenzwert):  |P_closed - P_mp(p+1e-12)| < 1e-8 (P ist stetig; P(p+t)~P(p)-t) an p = 5, 7, 23
 V6  t -> 1: P_float(p+0,999999) stetig gegen sigma(p+1)-p-2 - kleiner Cutoffterm:  |P - (sigma(p+1)-p-2)| < 1e-6 bei kappa = 6,4 (p+1) ln p, p = 5, 7, 11, 13
 M1  Mutant: D weggelassen -> V1 muss an p=5, C=1 FALLEN (D ist dort > 1e-12 noetig)
 M2  Mutant: E-Faktor om durch 1/2 ersetzt (Randgewicht konstant) -> V1 muss fallen
 M3  Mutant: Gewicht i durch 1 ersetzt (P_tau-artig) -> V2 muss fallen
Ausgabe: logs/E7790_gate_zerlegung__20261006_1250.txt ; Exitcode 0 nur wenn alles wie vorgesehen.
"""
import math
import os
import sys
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1195lib as L

CELLS_P = [5, 7, 23, 101]
CELLS_C = [1.0, 3.1, 6.4]
CELLS_T = [0.18, 0.3, 0.6, 0.9, 0.999]


def main():
    lines = []
    ok = True

    def out(s=""):
        print(s)
        lines.append(s)

    v1 = v2 = 0.0
    v3 = True
    v4 = True
    for p in CELLS_P:
        for C in CELLS_C:
            kap = C * (p + 1) * math.log(p)
            t = np.array(CELLS_T)
            d = L.parts(p, t, kap)
            rec = d["Pinf"] - d["om"] * d["E"] - d["D"] + d["H"]
            v1 = max(v1, float(np.max(np.abs(rec - d["P"]))) / p)
            for k, tt in enumerate(CELLS_T):
                pm = float(L.P_mp(p, tt, kap))
                v2 = max(v2, abs(pm - float(d["P"][k])))
            v3 &= bool(np.all(d["Lam"] <= d["P"] + 1e-12) and np.all(d["H"] >= 0) and np.all(d["D"] >= 0)
                       and np.all(d["om"] > 0) and np.all(d["om"] <= 0.5 + 1e-12))
            l1 = L.parts(p, t, kap)["Lam"]
            l2 = L.parts(p, t, 2 * kap)["Lam"]
            l4 = L.parts(p, t, 4 * kap)["Lam"]
            v4 &= bool(np.all(l1 <= l2 + 1e-12) and np.all(l2 <= l4 + 1e-12))
    out(f"V1 Zerlegung: max |rec-P|/p = {v1:.2e}  -> {'ok' if v1 < 1e-9 else 'ROT'}")
    out(f"V2 float gegen mpmath-Definition: max |dP| = {v2:.2e} -> {'ok' if v2 < 1e-8 else 'ROT'}")
    out(f"V3 Lam<=P, H>=0, D>=0, om in (0,1/2]: {'ok' if v3 else 'ROT'}")
    out(f"V4 Lam monoton in kappa (x1, x2, x4): {'ok' if v4 else 'ROT'}")
    ok &= v1 < 1e-9 and v2 < 1e-8 and v3 and v4
    v5 = 0.0
    for p in (5, 7, 23):
        kap = 3.1 * (p + 1) * math.log(p)
        v5 = max(v5, abs(L.P_at_p_closed(p, kap) - float(L.P_mp(p, mp.mpf("1e-12"), kap))))
    out(f"V5 P(p;kappa) geschlossene Form gegen mpmath bei t=1e-12: {v5:.2e} -> {'ok' if v5 < 1e-8 else 'ROT'}")
    ok &= v5 < 1e-8
    v6 = 0.0
    for p in (5, 7, 11, 13):
        kap = 6.4 * (p + 1) * math.log(p)
        sig = sum(i for i in range(1, p + 2) if (p + 1) % i == 0)
        v6 = max(v6, abs(float(L.parts(p, np.array([1 - 1e-9]), kap)["P"][0]) - (sig - p - 2)))
    out(f"V6 t -> 1: |P - (sigma(p+1)-p-2)| max = {v6:.2e} -> {'ok' if v6 < 1e-6 else 'ROT'}")
    ok &= v6 < 1e-6
    # Mutanten
    p, C = 5, 1.0
    kap = C * (p + 1) * math.log(p)
    t = np.array(CELLS_T)
    d = L.parts(p, t, kap)
    m1 = float(np.max(np.abs(d["Pinf"] - d["om"] * d["E"] + d["H"] - d["P"])))
    m2 = float(np.max(np.abs(d["Pinf"] - 0.5 * d["E"] - d["D"] + d["H"] - d["P"])))
    out(f"M1 (ohne D): Abweichung {m1:.2e} -> {'ok (faellt)' if m1 > 1e-9 else 'ROT (Detektor blind)'}")
    out(f"M2 (om=1/2): Abweichung {m2:.2e} -> {'ok (faellt)' if m2 > 1e-9 else 'ROT (Detektor blind)'}")
    # M3: Gewicht 1 statt i
    def p_tau_like(p, tt, kappa):
        x = p + tt
        s = 0.0
        for i in range(2, p + 10):
            u = i / (x + 1)
            ph = 1.0 / (1.0 + math.exp(2 * kappa * (u - 1)))
            F = math.sin(math.pi * x) ** 2 / math.sin(math.pi * x / i) ** 2
            s += ph * F / i ** 2
        return s - 1.0
    m3 = max(abs(p_tau_like(p, tt, kap) - float(d["P"][k])) for k, tt in enumerate(CELLS_T))
    out(f"M3 (Gewicht 1/i^2 statt i): Abweichung {m3:.2e} -> {'ok (faellt)' if m3 > 1e-6 else 'ROT'}")
    ok &= m1 > 1e-9 and m2 > 1e-9 and m3 > 1e-6
    out("GATE " + ("GRUEN" if ok else "ROT"))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "E7790_gate_zerlegung__20261006_1250.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
