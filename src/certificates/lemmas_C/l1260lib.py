"""T36-F-1260 Bibliothek: die zwei Hilfsabschaetzungen des API-Papers (Anhang C, p >= 23), ausgeschrieben und nachgerechnet.

Objekt (C4): P_sigma(x;kappa) = sum_{i>=2} phi_kappa(i/(x+1)) F(x,i)/i - x (API Def. 8.1), ENDLICHES kappa, p Primzahl >= 23
(Klassen p = 1, 5 mod 6), x = p + t, t in [0, 9/50], kappa >= C (p+1) ln p, C = 3,25.  Nicht P_inf, nicht P_tau, nicht Verm. 8.3.

Lemma K (Klassengewichte, B2 der Gegenlese T35-V-1091):
    G(q,rho) = { d/m : m | gcd(d,6), 2 <= d/m <= q+1 },  d = q - rho   (garantierte Teiler)
    c_a(q,rho) = (1/q) sum_{i in G, i != q+1} i,    c_b(q,rho) = (q+1)/q falls q+1 in G, sonst 0.
    FUER q > R + 12 (R = 40) haengt die Multiplikatorenmenge M(q,rho) nur von (q mod 6, rho) ab:
        M = { m | gcd(r - rho, 6) : m >= 2 oder rho >= 1 },  K_r(rho) = sum_{m in M} 1/m,
        c_a(q,rho) = (1 - rho/q) K_r(rho),   c_b(q,-1) = 1 + 1/q, sonst c_b = 0.
Lemma E (Restschranke, B3): siehe eps_formula.

Exakte Arithmetik mit fractions.Fraction; Kugelarithmetik mit python-flint (arb), Praezision 200 bit.
"""
import math
from fractions import Fraction
from flint import arb, ctx

ctx.prec = 200
PI = arb.pi()
R = 40
M6 = (1, 2, 3, 6)


# ------------------------------------------------------------------ Lemma K (exakt)
def G_set(q, rho, M=6):
    """Garantierte Teiler i = d/m, m | gcd(d, M), 2 <= i <= q+1 (direkt aus der Definition, keine Abkuerzung)."""
    d = q - rho
    if d <= 0:
        return []
    g = math.gcd(d, M)
    out = []
    for m in range(1, g + 1):
        if g % m == 0:
            i = d // m
            if 2 <= i <= q + 1:
                out.append(i)
    return out


def c_a_q(q, rho):
    return Fraction(sum(i for i in G_set(q, rho) if i != q + 1), q)


def c_b_q(q, rho):
    return Fraction(q + 1, q) if (q + 1) in G_set(q, rho) else Fraction(0)


def K_formula(r, rho, threshold_m1=True):
    """K_r(rho) = sum_{m | gcd(r-rho,6), m >= 2 oder rho >= 1} 1/m  (Formel des Lemmas)."""
    g = math.gcd(r - rho, 6)
    tot = Fraction(0)
    for m in M6:
        if g % m == 0 and (m >= 2 or rho >= 1 or not threshold_m1):
            tot += Fraction(1, m)
    return tot


def class_coeffs_exact(r, p_lo, scan_to=300, Rr=R, K_scale=Fraction(1)):
    """Exakte Klassenkoeffizienten (ohne das 10^9+7-Stellvertreterelement des Altskripts):
    c_a(rho) = min( min_{q = r mod 6, p_lo <= q <= scan_to} c_a(q,rho),  K_r(rho) fuer rho < 0 (Limes q -> oo) ),
    c_b(rho) = min( min_q c_b(q,rho),  1 fuer rho = -1 / 0 sonst (Limes) ).
    K_scale != 1 ist nur fuer Mutanten im Gate."""
    qs = [q for q in range(p_lo, scan_to + 1) if q % 6 == r % 6]
    out = {}
    for rho in range(-Rr, Rr + 1):
        if rho == 0:
            continue
        ca = min(c_a_q(q, rho) for q in qs)
        cb = min(c_b_q(q, rho) for q in qs)
        if rho < 0:
            ca = min(ca, K_formula(r, rho) * K_scale)
        cb_lim = Fraction(1) if rho == -1 else Fraction(0)
        cb = min(cb, cb_lim)
        out[rho] = (ca, cb)
    return out


def check_structure(r, q_from, q_to, Rr=R, K_scale=Fraction(1)):
    """Zaehlt Verletzungen der Formeln c_a(q,rho) = (1-rho/q) K_r(rho), c_b(q,-1) = 1+1/q, c_b(q,rho)=0 sonst, fuer ALLE ganzen q = r (6)
    in [q_from, q_to] und 0 < |rho| <= Rr (exakte Bruchrechnung, Definition vs. Formel)."""
    bad = 0
    n = 0
    first = None
    for q in range(q_from, q_to + 1):
        if q % 6 != r % 6:
            continue
        for rho in range(-Rr, Rr + 1):
            if rho == 0:
                continue
            n += 1
            K = K_formula(r, rho) * K_scale
            ca_f = Fraction(q - rho, q) * K
            cb_f = Fraction(q + 1, q) if rho == -1 else Fraction(0)
            if c_a_q(q, rho) != ca_f or c_b_q(q, rho) != cb_f:
                bad += 1
                if first is None:
                    first = (q, rho)
    return bad, n, first


# ------------------------------------------------------------------ Lemma E (Arb)
def eps_formula(p, C=Fraction(13, 4), reps=None):
    """eps(p) = |B_rest|-Schranke / p, EIGENE Implementierung nach dem Lemma (nicht aus p1198uni importiert):
    ell = 2 C ln p,  Q1 = ell^2 + 4 sqrt2 pi ell + 2 ell/(p+1),
    z1 = ell (p+2)/(p+1),  Q2 = z1^2 + 2 ell (p+2)/(p+1)^2 + 4 sqrt2 pi z1,
    expo = ell (p+1)(41/50)/(p+1+9/50),
    eps(p) = e^{-ell} Q1/(1 - e^{-ell}) + 2 (p+2)/p * e^{-expo} * Q2.
    Rueckgabe: (eps, Teil1, Teil2) als arb."""
    p = arb(p)
    C = arb(C.numerator) / C.denominator if isinstance(C, Fraction) else arb(C)
    ell = 2 * C * p.log()
    c = 4 * arb(2).sqrt() * PI
    Q1 = ell * ell + c * ell + 2 * ell / (p + 1)
    z1 = ell * (p + 2) / (p + 1)
    Q2 = z1 * z1 + 2 * ell * (p + 2) / ((p + 1) ** 2) + c * z1
    expo = ell * (p + 1) * (arb(41) / 50) / (p + 1 + arb(9) / 50)
    e1 = (-ell).exp() * Q1 / (1 - (-ell).exp())
    e2 = 2 * (p + 2) / p * (-expo).exp() * Q2
    return e1 + e2, e1, e2


def sigma(s):
    return 1 / (1 + (-s).exp())
