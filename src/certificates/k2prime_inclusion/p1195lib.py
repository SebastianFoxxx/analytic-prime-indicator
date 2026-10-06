"""T36-F-1195 Bibliothek (Karte K2', Programm P).  EIGENER Code, kein Import aus p1021lib/p1175lib.

Objekt (C4): P_sigma(x;kappa) = sum_{i>=2} phi_kappa(i/(x+1)) F(x,i)/i - x   (API-Paper Def. 8.1, Gewichte i),
   F(x,i) = sin^2(pi x)/sin^2(pi x/i),  phi_kappa(u) = 1/(1 + exp(2 kappa (u-1))),  ENDLICHES kappa.
   NICHT P_tau (Gewichte 1/i), NICHT P_inf (kappa = inf).  x = p + t, p ungerade Primzahl, 0 < t < 1.
f_i(p+t) := F/i^2 = (sin(pi t)/(i sin(pi (p+t)/i)))^2   (sin^2(pi (p+t)) = sin^2(pi t)); Term_i = i phi_i f_i.
Zerlegung (exakt, t in (0,1)):  P_sigma = P_inf - (1-phi_{p+1}) E - D + H,
   E = (p+1) f_{p+1},  D = sum_{2<=i<=p} i (1-phi_i) f_i  >= 0,  H = sum_{i>=p+2} i phi_i f_i >= 0,
   P_inf = sum_{2<=i<=p+1} i f_i - x.
Untere Schranke  Lam(p,t;kappa) = sum_{2<=i<=p+1} i phi_i f_i - x  <=  P_sigma  (H >= 0), wachsend in kappa.
"""
import math
import numpy as np
import mpmath as mp


def sinc(y):
    y = np.asarray(y, dtype=float)
    out = np.ones_like(y)
    m = np.abs(y) > 1e-12
    out[m] = np.sin(np.pi * y[m]) / (np.pi * y[m])
    return out


def f_i(p, i, t):
    """f_i(p+t), t Array in (0,1]; stabil bei t -> 1 fuer i | p+1 (s-Form), sonst direkt."""
    t = np.asarray(t, dtype=float)
    if (p + 1) % i == 0:
        s = 1.0 - t
        r = sinc(s) / sinc(s / i)
        return r * r
    if p % i == 0:
        # i | p: Fejer-Trigonometrie-Polynom, keine Ausloeschung bei t -> 0 (Instrumentfix nach E7793 Lauf 2, C8)
        k = np.arange(-(i - 1), i)
        w = (1.0 - np.abs(k) / i) / i
        return (w[None, :] * np.cos(2.0 * np.pi * k[None, :] * t[:, None] / i)).sum(axis=1)
    m = p % i  # exakte ganzzahlige Reduktion statt sin(pi (p+t)/i) mit grossem Argument
    r = np.sin(np.pi * t) / (i * np.sin(np.pi * (m + t) / i))
    return r * r


def phi(kappa, u):
    e = 2.0 * kappa * (np.asarray(u, dtype=float) - 1.0)
    return 1.0 / (1.0 + np.exp(np.clip(e, -700, 700)))


def one_minus_phi(kappa, u):
    e = -2.0 * kappa * (np.asarray(u, dtype=float) - 1.0)
    return 1.0 / (1.0 + np.exp(np.clip(e, -700, 700)))


def imax_for(p, kappa, tmax=1.0, tail=50.0):
    a = 2.0 * kappa / (p + 1.0 + tmax)
    return int(math.ceil(p + 2 + tail / a)) + 3


def parts(p, t, kappa, imax=None):
    """Dict mit P_inf, E, omega (=1-phi_{p+1}), D, H, Lam, P  (alle fuer t-Array in [0,18; 1])."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if imax is None:
        imax = imax_for(p, kappa)
    x1 = p + 1.0 + t
    Pinf = -(p + t)
    D = np.zeros_like(t)
    H = np.zeros_like(t)
    Lam_sum = np.zeros_like(t)
    E = np.zeros_like(t)
    om = np.zeros_like(t)
    for i in range(2, imax + 1):
        f = f_i(p, i, t)
        u = i / x1
        if i <= p + 1:
            Pinf = Pinf + i * f
            omi = one_minus_phi(kappa, u)
            Lam_sum = Lam_sum + i * (1.0 - omi) * f
            if i <= p:
                D = D + i * omi * f
            else:
                E = (p + 1.0) * f
                om = omi
        else:
            H = H + i * phi(kappa, u) * f
    Lam = Lam_sum - (p + t)
    P = Lam + H
    return dict(Pinf=Pinf, E=E, om=om, D=D, H=H, Lam=Lam, P=P)


def P_mp(p, t, kappa, dps=40, imax=None):
    """Direkte Definition in mpmath (nur Gates)."""
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        t = mp.mpf(t)
        kappa = mp.mpf(kappa)
        x = p + t
        if imax is None:
            imax = int(p + 2 + 60 / (2 * float(kappa) / (p + 2))) + 5
        s = mp.mpf(0)
        for i in range(2, imax + 1):
            u = mp.mpf(i) / (x + 1)
            ph = 1 / (1 + mp.e ** (2 * kappa * (u - 1)))
            F = mp.sin(mp.pi * x) ** 2 / mp.sin(mp.pi * x / i) ** 2
            s += ph * F / i
        return s - x
    finally:
        mp.mp.dps = old


def P_at_p_closed(p, kappa):
    """P_sigma(p;kappa) = -p/(1+exp(2 kappa/(p+1)))  (nur i = p traegt, Kap. 17 §17.7.2 (2))."""
    return -p / (1.0 + math.exp(min(700.0, 2.0 * kappa / (p + 1.0))))


def primes_upto(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for k in range(2, int(n ** 0.5) + 1):
        if s[k]:
            s[k * k::k] = bytearray(len(s[k * k::k]))
    return [k for k in range(n + 1) if s[k]]
