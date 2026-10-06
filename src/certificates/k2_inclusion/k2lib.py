# -*- coding: utf-8 -*-
"""k2lib.py - eigener Code (Loop T36-F-1251, sub-f1251): Kugelarithmetik (python-flint/Arb)
fuer die Schranke L_R(p,t) des Einschluss-Satzes K2 (T36-F-993-T1) der Limesfunktion P_inf.

Objekt (C4): P_inf(x) = sum_{2<=i<x+1} sin^2(pi x)/(i sin^2(pi x/i)) - x   (kappa = inf, API-Paper).
Nicht: P_sigma bei endlichem kappa, P_tau, F-sharp.

Die Formeln werden hier aus dem WORTLAUT des Satzes (docs/reports/w101a_993/beweis_K2_einschluss.md
Lemma 1-4) neu geschrieben, NICHT aus code/T36_F993 oder code/T35_V945 importiert.

  g_rho(t)  = sin^2(pi t) / (pi^2 (rho+t)^2)
  c_inf(rho): rho=-1: 1/2 ; rho>=1: 1/2 (ungerade) / 0 (gerade) ; rho<=-2: -1/2 (ungerade) / -1 (gerade)
  T_R(t)    = sin^2(pi t)/pi^2 * 2R/(R^2-t^2)
  L_R(p,t)  = sum_{0<|rho|<=R} c_inf(rho) g_rho(t) - 3/(2p) sum_{rho=1}^R rho g_rho(t) - T_R(t) - t/p
"""
from flint import arb, ctx, fmpq, arb_series

ctx.prec = 200


def PI():
    return arb.pi()


def cinf(rho):
    """Garantierte Gewichtsschranke (w_unter - p)/p im Grenzwert p -> inf (Lemma 2)."""
    if rho == -1:
        return arb(1) / 2
    if rho >= 1:
        return arb(1) / 2 if rho % 2 == 1 else arb(0)
    # rho <= -2
    return -arb(1) / 2 if rho % 2 != 0 else arb(-1)


def sinc2_series(s, N=44):
    """sinc_pi(s)^2 fuer einen Ball s >= 0 mit s <= 0.82 (auch 0 enthaltend), Reihe
    sinc_pi(s) = sum (-1)^k (pi s)^(2k)/(2k+1)!  mit Restglied |naechster Term| (alternierend, fallend,
    weil (pi s)^2 <= 6.7 < 6*... ab k=0: Quotient (pi s)^2/((2k+2)(2k+3)) <= 6.7/6 -> ab k=1 <1; wir
    nehmen N so gross, dass der Rest klein ist, und schaetzen den Rest durch den Betrag des ersten
    ausgelassenen Terms mal 2 (konservativ: der Quotient der ersten Terme kann > 1 sein, ab k >= 2 < 0.34)."""
    x = PI() * s
    y = x * x
    # Horner in y
    # Term_k = (-y)^k / (2k+1)!
    fac = [arb(1)]
    for k in range(1, 2 * N + 4):
        fac.append(fac[-1] * k)
    tot = arb(0)
    ypow = arb(1)
    for k in range(N + 1):
        tot += ((-1) ** k) * ypow / fac[2 * k + 1]
        ypow = ypow * y
    # Rest: Terme k >= N+1, Quotient <= y/((2k+2)(2k+3)) <= 6.7/(2*45*... ) << 1/2: Rest <= 2*|Term_{N+1}|
    term = ypow / fac[2 * (N + 1) + 1]  # (y)^(N+1)/(2N+3)!
    rest = abs(term) * 2
    tot = tot + arb(0, rest.upper())
    return tot * tot


def g(rho, t):
    """g_rho(t) fuer Ball t in [0,18; 1]. rho=-1 in s=1-t Form (keine Division durch 0)."""
    if rho == -1:
        s = arb(1) - t
        if s.upper() <= arb("0.30"):
            return sinc2_series(s)
        st = (PI() * s).sin()  # s >= ... nicht 0
        return st * st / ((PI() * s) ** 2)
    st = t.sin_pi()
    return st * st / (PI() * PI() * (rho + t) ** 2)


def TR(R, t):
    st = t.sin_pi()
    return st * st / (PI() * PI()) * (2 * R) / (R * R - t * t)


def L(R, p, t, cfun=None, drop_half=False):
    """L_R(p,t). p darf arb (auch 'unendlich' -> Mutante p=None: p-Terme weglassen) sein.
    cfun: alternative c_inf (Mutanten/Gegenfamilien)."""
    cf = cfun or cinf
    tot = arb(0)
    s1 = arb(0)
    for rho in range(-R, R + 1):
        if rho == 0:
            continue
        gr = g(rho, t)
        tot += cf(rho) * gr
        if rho >= 1:
            s1 += rho * gr
    out = tot - TR(R, t)
    if p is None:  # Limes p -> inf
        return out
    return out - arb(3) / (2 * p) * s1 - t / p


def U_V(R, t):
    """Zerlegung L_R(p,t) = U(t) - V(t)/p mit U = sum c g - T_R, V = (3/2) S1 + t."""
    tot = arb(0)
    s1 = arb(0)
    for rho in range(-R, R + 1):
        if rho == 0:
            continue
        gr = g(rho, t)
        tot += cinf(rho) * gr
        if rho >= 1:
            s1 += rho * gr
    U = tot - TR(R, t)
    V = arb(3) / 2 * s1 + t
    return U, V


def box_from(a, b):
    """Ball, der [a,b] (a,b arb mit exakter Mitte) einschliesst."""
    return a.union(b)


def lo_hi(x):
    return x.lower(), x.upper()


def midpoint_exact(lo, hi):
    return ((lo + hi) / 2).mid()


# ------------------------------------------------------------------ Taylor-Modell (andere Mechanik)
def gs(rho, t_ser):
    st = t_ser.sin_pi()
    return st * st / (PI() * PI() * (t_ser + rho) ** 2)


def TRs(R, t_ser):
    st = t_ser.sin_pi()
    return st * st / (PI() * PI()) * (2 * R) / (R * R - t_ser * t_ser)


def Fseries(R, p, tball, order=3):
    """Taylor-Reihe von L_R(p, .) um den Ball tball (Konstantterm = tball, Koeffizient x^1 = 1)."""
    t = arb_series([tball, 1], prec=order)
    tot = arb_series([0], prec=order)
    s1 = arb_series([0], prec=order)
    for rho in range(-R, R + 1):
        if rho == 0:
            continue
        gr = gs(rho, t)
        tot = tot + cinf(rho) * gr
        if rho >= 1:
            s1 = s1 + rho * gr
    out = tot - TRs(R, t)
    return out - arb(3) / (2 * p) * s1 - t / p


# ------------------------------------------------------------------ Zertifikats-Maschinen
def certify_adaptive(R, p, a, b, thr, fun=None, max_boxes=400000, min_w=None):
    """Adaptive Bisektion auf [a,b]: jeder Kasten muss  L(R,p,Kasten) - thr > 0  (sicher) erfuellen.
    a,b: arb (a<=b; die Enden werden mit lower()/upper() nach aussen genommen -> ueberdeckt [a,b] sicher).
    Rueckgabe: dict(ok, boxes, splits, min_lower, fail_boxes)."""
    f = fun or (lambda t: L(R, p, t))
    lo0, hi0 = a.lower(), b.upper()
    stack = [(lo0, hi0)]
    boxes = 0
    splits = 0
    min_lower = None
    fails = []
    while stack:
        lo, hi = stack.pop()
        t = lo.union(hi)
        val = f(t)
        boxes += 1
        if boxes > max_boxes:
            fails.append((lo, hi, "max_boxes"))
            break
        if (val - thr) > 0:
            lw = val.lower()
            if min_lower is None or lw < min_lower:
                min_lower = lw
            continue
        # nicht entschieden: teilen (bis Breite < 2^-40, dann Fehlkasten)
        if (hi - lo) < (min_w if min_w is not None else arb(2) ** -40):
            fails.append((lo, hi, val.str(10)))
            continue
        m = ((lo + hi) / 2).mid()
        splits += 1
        stack.append((lo, m))
        stack.append((m, hi))
    return dict(ok=(len(fails) == 0), boxes=boxes, splits=splits, min_lower=min_lower, fail_boxes=fails)


def certify_taylor(R, p, a, b, thr, h0=arb("0.01"), max_boxes=100000, min_w=None):
    """Zweite Mechanik: Taylor-Modell 2. Ordnung je Kasten (Mittelpunkt exakt, f'' per Ball ueber dem Kasten):
       f(t) >= f(m) - |f'(m)| h - sup|c2| h^2   mit c2 = f''/2 auf dem Kasten.
       Nur fuer Kasten mit t <= 0.92 (kein s=0-Problem in t-Form)."""
    lo0, hi0 = a.lower(), b.upper()
    stack = [(lo0, hi0)]
    boxes = 0
    splits = 0
    min_lower = None
    fails = []
    p = arb(p)
    while stack:
        lo, hi = stack.pop()
        boxes += 1
        if boxes > max_boxes:
            fails.append((lo, hi, "max_boxes"))
            break
        # vorab auf Breite <= h0 zuschneiden
        if (hi - lo) > h0:
            m = ((lo + hi) / 2).mid()
            splits += 1
            stack.append((lo, m))
            stack.append((m, hi))
            continue
        m = ((lo + hi) / 2).mid()
        hh = ((hi - lo) / 2).upper()
        sm = Fseries(R, p, m)
        sb = Fseries(R, p, lo.union(hi))
        f0 = sm[0]
        f1 = sm[1]
        c2 = sb[2]
        c2max = abs(c2).upper()
        lowb = f0 - abs(f1).upper() * hh - c2max * hh * hh
        if (lowb - thr) > 0:
            lw = lowb.lower()
            if min_lower is None or lw < min_lower:
                min_lower = lw
            continue
        if (hi - lo) < (min_w if min_w is not None else arb(2) ** -30):
            fails.append((lo, hi, lowb.str(10)))
            continue
        mm = m
        splits += 1
        stack.append((lo, mm))
        stack.append((mm, hi))
    return dict(ok=(len(fails) == 0), boxes=boxes, splits=splits, min_lower=min_lower, fail_boxes=fails)
