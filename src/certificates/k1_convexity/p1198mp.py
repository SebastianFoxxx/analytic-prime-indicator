"""T36-F-1198: unabhaengige Referenzrechnung (mpmath, direkt aus der Definition, KEIN Import aus p1198lib).
P_sigma(x;kappa) = sum_{i>=2} phi(i/(x+1)) F(x,i)/i - x ; phi(u) = 1/(1+exp(2 kappa (u-1))) ; F = sin^2(pi x)/sin^2(pi x/i)."""
import mpmath as mp

mp.mp.dps = 40


def phi(i, x, kappa):
    return 1 / (1 + mp.exp(2 * kappa * (mp.mpf(i) / (x + 1) - 1)))


def f_i(i, x):
    return (mp.sin(mp.pi * x) / (i * mp.sin(mp.pi * x / i))) ** 2


def imax(p, kappa, tmax=0.2):
    return int(p + 2 + 60 * (p + 1.2) / (2 * kappa)) + 4


def P(x, p, kappa):
    x = mp.mpf(x)
    I = imax(p, kappa)
    s = mp.mpf(0)
    for i in range(2, I + 1):
        s += phi(i, x, kappa) * i * f_i(i, x)
    return s - x


def term_derivs(i, x, kappa):
    """(phi, phi', phi'', f, f', f'') bei x ueber mpmath.diff."""
    ph = [mp.diff(lambda y: phi(i, y, kappa), x, k) if k else phi(i, x, kappa) for k in range(3)]
    fi = [mp.diff(lambda y: f_i(i, y), x, k) if k else f_i(i, x) for k in range(3)]
    return ph, fi


def split_AB(p, t, kappa):
    """(A, B_edge, B_rest, P'') exakt aus den Einzelableitungen (Produktregel)."""
    x = mp.mpf(p) + mp.mpf(t)
    I = imax(p, kappa)
    A = mp.mpf(0)
    Be = mp.mpf(0)
    Br = mp.mpf(0)
    for i in range(2, I + 1):
        ph, fi = term_derivs(i, x, kappa)
        A += i * ph[0] * fi[2]
        b = i * (ph[2] * fi[0] + 2 * ph[1] * fi[1])
        if i == p + 1:
            Be += b
        else:
            Br += b
    return A, Be, Br, A + Be + Br


def Pdd_direct(p, t, kappa):
    x = mp.mpf(p) + mp.mpf(t)
    return mp.diff(lambda y: P(y, p, kappa), x, 2)
