"""T36-F-1198 (R1b): p-uniforme Fassung des K1'-Zertifikats fuer ALLE Primzahlen p >= 23 (Kugelarithmetik, python-flint).

Teil P23: Restklassen p mod 6, garantierte Teiler d/m (m | gcd(d,6)), Randsummand exakt in (t, s), u = 1/(p+1) als Ball, eps p-uniform.
Baut auf p1198lib auf (g0pp, gpp_rho, sig, nfun, mfun, s_grid, s_boxes_lower).
"""
import math
from fractions import Fraction

from flint import arb

import p1198lib as L

PI = L.PI


def guaranteed_terms_q(q, rho, M=6):
    """Garantierte Teiler i = d/m (d = q - rho > 0, m | gcd(d, M)), 2 <= i <= q+1 (Lemma 4'; elementar; Lean fuer m in {1,2}: T36-F-996-T1)."""
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


def class_coeffs(r, M=6, p_lo=23, R=40, scan_to=300, parity_only=False):
    """p-uniforme Koeffizienten je Restklasse p = r mod M.  Rueckgabe: rho -> (c_a, c_b) als Fraction (exakt):
    c_a = min_q (1/q) sum_{i in G(q,rho), i <= q} i,  c_b = min_q [q+1 in G(q,rho)] (q+1)/q;
    Minimum ueber ALLE ganzen q in [p_lo, scan_to] der Klasse (Obermenge der Primzahlen) und den Limes (q = Klassenvertreter > 10^9)."""
    Mm = 2 if parity_only else M
    rr = 1 if parity_only else r
    qs = [q for q in range(p_lo, scan_to + 1) if q % Mm == rr % Mm]
    big = 10 ** 9 + 7
    while big % Mm != rr % Mm:
        big += 1
    qs = qs + [big]
    out = {}
    for rho in range(-R, R + 1):
        if rho == 0:
            continue
        ca = None
        cb = None
        for q in qs:
            terms = guaranteed_terms_q(q, rho, Mm)
            a = Fraction(sum(i for i in terms if i != q + 1), q)
            b = Fraction(q + 1, q) if (q + 1) in terms else Fraction(0)
            ca = a if ca is None else min(ca, a)
            cb = b if cb is None else min(cb, b)
        out[rho] = (ca, cb)
    return out


def _fr(x):
    return arb(x.numerator) / x.denominator


def A_coeffs_uniform(t, sigma0, coeffs):
    """A/p >= A0(t) + sigma(s) A1(t):  A0 = g0'' + sigma0 sum c_a g_rho'',  A1 = sum c_b g_rho''."""
    A0 = L.g0pp(t)
    A1 = arb(0)
    for rho, (ca, cb) in coeffs.items():
        if ca == 0 and cb == 0:
            continue
        g = L.gpp_rho(rho, t)
        if ca:
            A0 += sigma0 * _fr(ca) * g
        if cb:
            A1 += _fr(cb) * g
    return A0, A1


def edge_UV_u(t, u):
    """U(t,u), V(t,u) mit u = 1/(p+1) als Ball (p-uniform):  v = 1-t, theta = pi v u, rho = Sc(t)/(v sinc_pi(v u)), q = rho^2, a = 1/(1+t u),
    U = a^2 q,  V = a q [4 + 4 T1 + 4 (t/v) cos(theta)/sinc_pi(v u) - 2 t u/(1+t u)]   (Herleitung wie p1198lib.edge_UV)."""
    t = arb(t)
    v = 1 - t
    Sc = t.sinc_pi()
    sv = (v * u).sinc_pi()
    rho = Sc / (v * sv)
    q = rho * rho
    a = 1 / (1 + t * u)
    T1 = (PI * t).cos() / Sc - 1
    theta = PI * v * u
    U = a * a * q
    V = a * q * (4 + 4 * T1 + 4 * (t / v) * theta.cos() / sv - 2 * t * u / (1 + t * u))
    return U, V


def eps_uniform(C=3.25, p0=23):
    """p-uniforme Schranke E mit |B_rest|/p <= E fuer ALLE Primzahlen p >= p0, kappa >= C (p+1) ln p, t in [0, 9/50] (Fm = 1, elementar).
    i <= p (j = p+1-i >= 1):  term <= i p^(-2 C j) L2,  L2 = (2 C ln p)^2 + 4 C ln p/(p+1) + 2 (2 C ln p) 2 pi sqrt2  (Auswertung bei kappa_min).
    i = p+1+j' (j' >= 1):     term <= i exp(-2 C ln p (p+1)(j'-0.18)/(p+1.18)) L2'(j'),  Summe <= 2 * (j'=1)-Term (Quotient < 1/2, geprueft).
    Jeder Faktor ist Produkt aus p^-a ln^b p mit a ln p > b -> faellt in p: Maximum bei p0 (elementare Ableitung)."""
    p = arb(p0)
    C = arb(C)
    lnp = p.log()
    c = 2 * PI * arb(2).sqrt()
    two = 2 * C * lnp
    L2 = two * two + 4 * C * lnp / (p + 1) + 2 * two * c
    s1 = p * (p ** (-2 * C)) * L2 / (1 - p ** (-2 * C))
    i1 = p + 2
    z1 = two * (1 + 1 / (p + 1))
    L21 = z1 * z1 + 4 * C * lnp * i1 / (p + 1) ** 2 + 2 * z1 * c
    expo = 2 * C * lnp * (p + 1) * (arb(1) - arb(9) / 50) / (p + arb(1) + arb(9) / 50)
    t1 = i1 * (-expo).exp() * L21
    ratio = ((i1 + 1) / i1) ** 4 * (1 + 1 / (p + 1)) ** 2 * (-(2 * C * lnp * (p + 1) / (p + arb(1) + arb(9) / 50))).exp()
    ok_ratio = bool(ratio < 0.5)
    tot = (s1 + 2 * t1) / p
    return tot.upper(), {"ratio_ok": ok_ratio, "s_le_p": float(s1.upper()), "s_gt_p": float((2 * t1).upper()), "ratio": float(ratio.upper())}


def certify_uniform(r, C=3.25, p_lo=23, R=40, margin=0.0, grid=None, sb=None, tol_width=1e-6, first_split=16, mut=None,
                    u_split=(0.0, 1 / 100, 1 / 24), coeffs=None):
    """Intervallzertifikat p-uniform fuer ALLE Primzahlen p >= p_lo der Klasse p = r mod 6, kappa >= C (p+1) ln p, t in [0, 9/50]:
    P''/p >= A0 + sigma(s) A1 + ((p_lo+1)/p_lo) U m(s) + V n(s) - E   (Randform exakt in (t, s), u = 1/(p+1) in [0, 1/(p_lo+1)])."""
    mut = mut or {}
    if coeffs is None:
        coeffs = class_coeffs(r, p_lo=p_lo, R=mut.get("R", R), parity_only=mut.get("parity_only", False))
    sigma0 = L.sig(2 * arb(C) * arb(p_lo).log())
    E, einfo = eps_uniform(C, p_lo)
    if mut.get("eps_zero"):
        E = arb(0)
    if grid is None:
        grid = L.s_grid()
    if sb is None:
        sb = L.s_boxes_lower(grid)
    fac = arb(p_lo + 1) / p_lo
    stack = [(i / first_split * 0.18000000000000002, (i + 1) / first_split * 0.18000000000000002) for i in range(first_split)]
    done = []
    fails = []
    ub = [arb((a + b) / 2, (b - a) / 2 * 1.0000001) for a, b in zip(u_split[:-1], u_split[1:])]
    while stack:
        t0, t1 = stack.pop()
        t = arb((t0 + t1) / 2, (t1 - t0) / 2 * 1.0000001)
        A0, A1 = A_coeffs_uniform(t, sigma0, coeffs)
        A0l, A1l = A0.lower(), A1.lower()
        UVs = [edge_UV_u(t, uu) for uu in ub]
        Vl = min((V.lower() for U, V in UVs), key=lambda x: float(x.mid()))
        Uu = max((U.upper() for U, V in UVs), key=lambda x: float(x.mid()))
        if mut.get("flipV"):
            Vl = -Vl
        best = None
        # kappa >= C (p+1) ln p  <=>  s >= s_min(t) = 2 C ln(p) (p+1) t/(p+1+t) >= 2 C ln(p_lo) (p_lo+1) t0/(p_lo+1+t0)   (waechst in p und in t)
        s_lo_min = 2 * C * math.log(p_lo) * (p_lo + 1) * t0 / (p_lo + 1 + t0) * (1 - 1e-9)
        for k, (sl, nl, ml) in enumerate(sb):
            if mut.get("no_s_restrict") is None and k < len(grid) and grid[k][1] < s_lo_min:
                continue
            val = A0l + sl * A1l + fac * Uu * ml + Vl * nl - E
            lo = val.lower()
            if best is None or lo < best:
                best = lo
        if bool(best > margin):
            done.append((t0, t1, best))
        else:
            if t1 - t0 < tol_width:
                fails.append((t0, t1, best))
            else:
                m = (t0 + t1) / 2
                stack.append((t0, m))
                stack.append((m, t1))
    min_val = min((d[2] for d in done), default=None)
    arg = min(done, key=lambda d: d[2]) if done else None
    return {"ok": (not fails) and bool(done), "min_over_p": min_val, "argmin_box": arg, "boxes": done, "fails": fails, "E": E, "einfo": einfo,
            "n_boxes": len(done), "coeffs": coeffs, "sigma0": sigma0}
