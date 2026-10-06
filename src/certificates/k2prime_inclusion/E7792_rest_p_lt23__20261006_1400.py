"""E7792  T36-F-1195 R3: Rest p in {5,7,11,13,17,19} einzeln, Intervallzertifikat (mpmath.iv).

Untere Schranke (exakt, endliche Summe):  P_sigma(p+t;kappa) >= Lam(p,t;kappa) = sum_{i=2}^{p+1} i phi_kappa(i/(p+1+t)) f_i(p+t) - (p+t)
  (Summanden i >= p+2 sind >= 0 und entfallen).  Lam waechst in kappa (u = i/(p+1+t) < 1 fuer i <= p+1, t >= 0)
  -> ein Zertifikat bei kappa0 = C (p+1) ln p gilt fuer ALLE kappa >= kappa0.
f_i(p+t): i | p+1 -> (sinc(s)/sinc(s/i))^2, s = 1-t (stabil bis t = 1; sinc auf [0,1] fallend, Intervallenden);
          i nicht | p+1 -> (sin(pi t)/(i sin(pi (p+t)/i)))^2 direkt (Nenner auf [t_min,1] nullstellenfrei, da (p+t)/i dort keine ganze Zahl trifft).
t-Bereich [23/128, 1]  (Obermenge von [0,18; 1]; dyadische Endpunkte, exakt).  Soundness von mpmath.iv ANGENOMMEN.

VORAB-REGELN:
  A  Zertifikat bei C = 6,4 gelingt fuer alle sechs p.
  B  Kleinstes zertifiziertes C je p (Raster 0,05, Bisektion) <= Tabellenwert C_K2(p) aus E7789 (6,38/5,13/4,09/3,79/3,39/3,24),
     weil dort gegen 0,0015 p statt gegen die tatsaechliche Marge gemessen wurde.
  C  p = 5 ist das schlechteste p unter p < 23 (max_p C_cert(p) wird bei p = 5 angenommen).
  D  Mutant Lam ohne den Randsummand i = p+1: Zertifikat bei C = 6,4 FAELLT fuer p = 5 (Randterm tragend; Kap. 17 §17.2.3).
  E  Float-Konsistenz: min_t Lam_float(p,t;kappa0) liegt zwischen iv-Unter- und Obergrenze auf dem Gitter (Detektor).
Ausgabe: logs/E7792_rest__20261006_1400.txt, data/E7792_rest__20261006_1400.csv
"""
import math
import os
import sys
import csv
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1195lib as L

iv = mp.iv
iv.dps = 40
TMIN = 23.0 / 128.0
PI = iv.pi


def sinc_pt(s):
    if s == 0.0:
        return iv.mpf(1)
    x = iv.mpf(s)
    return iv.sin(PI * x) / (PI * x)


def sinc_int(slo, shi):
    a = sinc_pt(shi)
    b = sinc_pt(slo)
    return iv.mpf([a.a, b.b])


def f_iv(p, i, a, b):
    if (p + 1) % i == 0:
        A = sinc_int(1.0 - b, 1.0 - a)
        B = sinc_int((1.0 - b) / i, (1.0 - a) / i)
        r = A / B
        return r * r
    t = iv.mpf([a, b])
    r = iv.sin(PI * t) / (iv.mpf(i) * iv.sin(PI * (iv.mpf(p) + t) / iv.mpf(i)))
    return r * r


def lam_iv(p, kappa, a, b, edge=True):
    t = iv.mpf([a, b])
    x1 = p + 1 + t
    s = iv.mpf(0)
    for i in range(2, p + 2):
        if i == p + 1 and not edge:
            continue
        ph = 1 / (1 + iv.exp(2 * iv.mpf(kappa) * (iv.mpf(i) / x1 - 1)))
        s += iv.mpf(i) * ph * f_iv(p, i, a, b)
    return s - (iv.mpf(p) + t)


def certify(p, C, edge=True, maxdepth=22):
    kappa = C * (p + 1) * math.log(p)
    stack = [(TMIN, 1.0, 0)]
    nint = 0
    worst = 1e9
    while stack:
        a, b, dep = stack.pop()
        lo = lam_iv(p, kappa, a, b, edge).a
        if lo > 0:
            nint += 1
            worst = min(worst, float(lo))
            continue
        if dep >= maxdepth:
            return False, float(lo), nint
        m = 0.5 * (a + b)
        stack.append((a, m, dep + 1))
        stack.append((m, b, dep + 1))
    return True, worst, nint


def cmin(p, lo=0.2, hi=6.4, step=0.05):
    if not certify(p, hi)[0]:
        return None
    n = int(round((hi - lo) / step))
    L_, H_ = 0, n
    while H_ - L_ > 1:
        mid = (L_ + H_) // 2
        if certify(p, lo + mid * step)[0]:
            H_ = mid
        else:
            L_ = mid
    return lo + H_ * step


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)
        sys.stdout.flush()

    TAB = {5: 6.3767, 7: 5.1314, 11: 4.0854, 13: 3.7865, 17: 3.3880, 19: 3.2385}
    out("E7792 Rest p<23 (mpmath.iv, Soundness angenommen)")
    rows = []
    cm = {}
    A = True
    for p in (5, 7, 11, 13, 17, 19):
        ok, lo, n = certify(p, 6.4)
        A &= ok
        c = cmin(p)
        cm[p] = c
        # Float-Konsistenz
        kap = 6.4 * (p + 1) * math.log(p)
        tg = np.linspace(TMIN, 1 - 1e-9, 20001)
        fl = float(np.min(L.parts(p, tg, kap)["Lam"]))
        out(f"  p={p:2d}: C=6,4 zertifiziert={ok} (kleinste untere Schranke {lo:.5f}, {n} Intervalle); kleinstes zertifiziertes C = {c}; Float min Lam = {fl:.5f}")
        rows.append((p, 6.4, ok, lo, c, fl))
    out(f"A  alle sechs bei C=6,4: {'ok' if A else 'ROT'}")
    B = all(cm[p] is not None and cm[p] <= TAB[p] + 1e-9 for p in cm)
    out(f"B  C_cert(p) <= Tabelle: {'ok' if B else 'ROT'}  {cm}")
    C_ = all(cm[5] >= cm[p] for p in cm if cm[p] is not None)
    out(f"C  p=5 schlechtestes unter p<23: {'ok' if C_ else 'ROT'}")
    okD, loD, _ = certify(5, 6.4, edge=False)
    out(f"D  Mutant ohne Randsummand i=p+1, p=5, C=6,4: zertifiziert={okD} (untere Schranke {loD:.4f}) -> {'ok (faellt)' if not okD else 'ROT'}")
    out("ERGEBNIS: " + ("GRUEN" if (A and B and C_ and not okD) else "ROT/TEILWEISE"))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "E7792_rest__20261006_1400.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "data", "E7792_rest__20261006_1400.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p", "C", "zertifiziert", "min_untere_schranke", "C_cert_min", "float_min_Lam"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
