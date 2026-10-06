"""T36-F-1198 Bibliothek (Programm P, Karte K3'; Runde R1: K1' fuer p < 23 in Kugelarithmetik).

Objekt (C4): P_sigma(x;kappa) = sum_{i>=2} phi_kappa(i/(x+1)) F(x,i)/i - x  (API-Paper Def. 8.1, Gewichte i), ENDLICHES kappa,
  phi_kappa(u) = 1/(1+exp(2 kappa (u-1))),  x = p + t,  p in {5,7,11,13,17,19},  kappa >= C (p+1) ln p.
Nicht gemeint: P_inf, P_tau, Originalindikator, Vermutung 8.3.

Zerlegung (Produktregel, exakt):  P'' = A + B,  f_i = F/i^2 = sum_j g_{p+ij},  g_rho(t) = sin^2(pi t)/(pi^2 (rho+t)^2)
  A = sum_i i phi_i f_i'' = sum_rho W(rho,t) g_rho''(t),  W(rho,t) = sum_{i>=2, i | (p-rho)} i phi_i(t)
  B = sum_i i (phi_i'' f_i + 2 phi_i' f_i').
Untere Schranke (alle kappa >= kappa_min, alle t in [0, 0.18]):
  A >= A0(t) + sigma(s) A1(t)   mit  A0 = p g0'' + sigma0 sum_{rho != 0, |rho|<=R} D(rho) g_rho'' ,  A1 = (p+1) sum_{rho: (p+1)|(p-rho)} g_rho'',
        sigma0 = sigma(2 C ln p)  (phi_i >= sigma0 fuer i <= p),  s = 2 kappa t/(p+1+t)  (phi_{p+1} = sigma(s) EXAKT),
  B = B_edge + B_rest,  B_edge/(p+1) = U s^2 sigma''(s) + V s sigma'(s)  (exakt, U(t), V(t) geschlossen),  |B_rest| <= eps_p.
Alles in python-flint (arb), Kugelarithmetik; die Soundness der Bibliothek ist angenommen.
"""
import math
from flint import arb, ctx

ctx.prec = 160
PI = arb.pi()


def A(x):
    return arb(x)


# ------------------------------------------------------------------ g_rho'' (Kap. 17 Lemma 3; Lean T36-F-994-T1 fuer das Vorzeichen)
def gpp_rho(rho, t):
    """g_rho''(t) = H/(pi^2 y^4), y = rho + t, H = 2 pi^2 cos(2 pi t) y^2 - 4 pi sin(2 pi t) y + 6 sin^2(pi t); rho != 0 ganz."""
    y = arb(rho) + t
    c2 = (2 * PI * t).cos()
    s2 = (2 * PI * t).sin()
    s1 = (PI * t).sin()
    H = 2 * PI * PI * c2 * y * y - 4 * PI * s2 * y + 6 * s1 * s1
    return H / (PI * PI * y ** 4)


def g0pp(t, N=16):
    """g_0''(t) = sum_{k>=2} (-1)^{k+1} (2 pi)^{2k} (2k-2)(2k-3) t^{2k-4} / (2 pi^2 (2k)!), t in [0, 0.18] (Ball); Rest <= 2 * erster ausgelassener Term."""
    tot = arb(0)
    for k in range(2, N + 1):
        term = ((-1) ** (k + 1)) * (2 * PI) ** (2 * k) * (2 * k - 2) * (2 * k - 3) * t ** (2 * k - 4) / (2 * PI * PI * arb.fac_ui(2 * k))
        tot += term
    k = N + 1
    tmax = t.upper()
    nxt = ((2 * PI) ** (2 * k) * (2 * k - 2) * (2 * k - 3) * tmax ** (2 * k - 4) / (2 * PI * PI * arb.fac_ui(2 * k)))
    rem = (2 * nxt).upper()
    return tot + arb(0, rem)


# ------------------------------------------------------------------ Randsummand i = p+1
def edge_UV(p, t):
    """U(t), V(t) mit  B_edge/(p+1) = U s^2 sigma''(s) + V s sigma'(s),  s = 2 kappa t/(p+1+t).
    f_{p+1}(p+t) = t^2 q(t), q = rho^2, rho = pi Sc/((p+1) sin(theta)), Sc = sin(pi t)/(pi t), theta = pi (1-t)/(p+1);
    a = (p+1)/(p+1+t);  U = a^2 q;  V = a q [4 + 4 T1 + 4 (pi t/(p+1)) cot(theta) - 2 t/(p+1+t)],  T1 = t Sc'/Sc = pi t cot(pi t) - 1."""
    t = arb(t)
    n1 = arb(p + 1)
    x1 = n1 + t
    Sc = t.sinc_pi()
    theta = PI * (1 - t) / n1
    rho = PI * Sc / (n1 * theta.sin())
    q = rho * rho
    a = n1 / x1
    T1 = (PI * t).cos() / Sc - 1
    cot_th = theta.cos() / theta.sin()
    U = a * a * q
    V = a * q * (4 + 4 * T1 + 4 * (PI * t / n1) * cot_th - 2 * t / x1)
    return U, V


def sig(s):
    return 1 / (1 + (-s).exp())


def nfun(s):
    """n(s) = s sigma'(s) = s/(4 cosh^2(s/2))."""
    c = (s / 2).cosh()
    return s / (4 * c * c)


def mfun(s):
    """m(s) = s^2 sigma''(s) = - s^2 tanh(s/2)/(4 cosh^2(s/2))  (<= 0 fuer s >= 0)."""
    c = (s / 2).cosh()
    return -(s * s) * (s / 2).tanh() / (4 * c * c)


# ------------------------------------------------------------------ Teilergewichte
def divisor_data(p, R):
    """Liste (rho, D_le_p, ind_p1) fuer 0 < |rho| <= R:  d = p - rho;  D = sum_{i | d, 2<=i<=p} i (d = 0: alle i = 2..p);
    ind_p1 = 1, falls (p+1) | d (d = 0 inklusive)."""
    out = []
    for rho in range(-R, R + 1):
        if rho == 0:
            continue
        d = abs(p - rho)
        if d == 0:
            D = sum(range(2, p + 1))
            ind = 1
        else:
            D = sum(i for i in range(2, p + 1) if d % i == 0)
            ind = 1 if d % (p + 1) == 0 else 0
        if D or ind:
            out.append((rho, D, ind))
    return out


def A_coeffs(p, t, sigma0, data):
    """(A0(t), A1(t)) als Baelle:  A0 = p g0'' + sigma0 sum D g_rho'',  A1 = (p+1) sum_{ind} g_rho''."""
    A0 = p * g0pp(t)
    A1 = arb(0)
    for rho, D, ind in data:
        g = gpp_rho(rho, t)
        if D:
            A0 += sigma0 * D * g
        if ind:
            A1 += (p + 1) * g
    return A0, A1


# ------------------------------------------------------------------ Rest-Cutoff-Schranke eps_p
def eps_rest(p, kappa, I_extra=60):
    """Obere Schranke fuer |B_rest| = |sum_{i != p+1} i (phi_i'' f_i + 2 phi_i' f_i')| gueltig fuer ALLE kappa' >= kappa, t in [0, 0.18].
    Je i:  |term| <= i e^{-2 kappa m_i} [ Fm ( (2 kappa i/(p+1)^2)^2 + 4 kappa i/(p+1)^3 ) + 2 (2 kappa i/(p+1)^2) 2 pi sqrt2 sqrt(Fm) ],
    m_i = 1 - i/(p+1) (i <= p),  m_i = i/(p+1.18) - 1 (i >= p+2);  Fm = 1 (i <= p), sonst sin^2(0.18 pi)/(i^2 min sin^2(pi x/i)).
    Monotonie in kappa: jeder Summand ~ kappa^a e^{-2 m kappa}, a <= 2, faellt fuer kappa >= 1/m (geprueft). Rueckgabe: (eps, details)."""
    kappa = arb(kappa)
    n1 = arb(p + 1)
    n118 = arb(p) + arb(1) + arb(9) / 50
    T018 = arb(9) / 50
    s018 = (PI * T018).sin() ** 2
    total = arb(0)
    det = {}
    last_term = None
    mono_ok = True

    def term(i):
        nonlocal mono_ok
        if i <= p:
            m = 1 - arb(i) / n1
            Fm = arb(1)
        else:
            m = arb(i) / n118 - 1
            xa = PI * arb(p) / i
            xb = PI * (arb(p) + arb(9) / 50) / i
            smin = xa.sin().min(xb.sin()) if hasattr(xa.sin(), 'min') else xa.sin()
            Fm = s018 / (arb(i) ** 2 * smin * smin)
        if not (kappa > 1 / m):
            mono_ok = False
        zp = 2 * kappa * i / (n1 * n1)
        zpp = 4 * kappa * i / (n1 ** 3)
        c = 2 * PI * arb(2).sqrt()
        val = i * (-2 * kappa * m).exp() * (Fm * (zp * zp + zpp) + 2 * zp * c * Fm.sqrt())
        return val

    for i in list(range(2, p + 1)) + list(range(p + 2, p + 2 + I_extra)):
        v = term(i)
        total += v
        det[i] = v.upper()
        last_term = (i, v)
    # Rest ab i = p+2+I_extra: Verhaeltnis zweier aufeinanderfolgender Schranken pruefen (f=1 als Obergrenze der Fm-Formel gilt hier nicht,
    # wir benutzen die Formel mit f<=1 als grobe, groessere Schranke) -> Rest <= 2 * Schranke(i0) mit f<=1
    i0 = p + 2 + I_extra

    def term_f1(i):
        m = arb(i) / n118 - 1
        zp = 2 * kappa * i / (n1 * n1)
        zpp = 4 * kappa * i / (n1 ** 3)
        c = 2 * PI * arb(2).sqrt()
        return i * (-2 * kappa * m).exp() * ((zp * zp + zpp) + 2 * zp * c)

    r = (term_f1(i0 + 1) / term_f1(i0)).upper()
    ratio_ok = bool(r < 0.5)
    tail = 2 * term_f1(i0)
    total += tail
    return total.upper(), {"mono_ok": mono_ok, "ratio_ok": ratio_ok, "ratio": float(r), "tail": float(tail.upper()),
                           "max_term": max(det.values()) if det else None, "det": det}


def kappa_min(p, C):
    return arb(C) * (p + 1) * arb(p).log()


# ------------------------------------------------------------------ s-Gitter (nur von s abhaengige Faktoren)
def s_grid(s_max=40, fine_to=12, w_fine=0.005, w_coarse=0.05):
    """Liste von (lo, hi) als exakte dyadische/ rationale Endpunkte (Python-float exakt als Dualzahl) ; Rest [s_max, inf) separat."""
    ed = []
    k = 0
    x = 0.0
    while x < fine_to - 1e-12:
        ed.append(x)
        k += 1
        x = k * w_fine
    x = float(fine_to)
    k = 0
    while x < s_max - 1e-12:
        ed.append(x)
        k += 1
        x = fine_to + k * w_coarse
    ed.append(float(s_max))
    return list(zip(ed[:-1], ed[1:]))


def s_boxes_lower(grid, s_max=40):
    """Fuer jede Box die unteren Schranken (sigma_lo, n_lo, m_lo) als arb-Zahlen (m_lo <= 0)."""
    out = []
    for lo, hi in grid:
        s = arb((lo + hi) / 2, (hi - lo) / 2 * 1.0000001)  # Ball ueber [lo, hi] (Radius leicht vergroessert)
        out.append((sig(s).lower(), nfun(s).lower(), mfun(s).lower()))
    tail = (sig(arb(s_max)).lower(), arb(0), -(arb(s_max) ** 2 * (-arb(s_max)).exp()))
    out.append(tail)
    return out


def t_box_lower(p, C, sigma0, data, t0, t1, mut=None):
    """Untere Schranken (A0l, A1l, Vl, Uu) fuer die t-Box [t0, t1] (Python-floats bzw. arb)."""
    t = arb((t0 + t1) / 2, (t1 - t0) / 2 * 1.0000001) if not isinstance(t0, arb) else arb(0)
    A0, A1 = A_coeffs(p, t, sigma0, data)
    U, V = edge_UV(p, t)
    if mut and mut.get("flipV"):
        V = -V
    return A0.lower(), A1.lower(), V.lower(), U.upper()


def certify(p, C, R=600, mut=None, t_hi_float=0.18000000000000002, tol_width=1e-6, verbose=False, grid=None, sb=None, t_lo_float=0.0,
            first_split=1, margin=0.0):
    """Intervallzertifikat: fuer alle kappa >= C (p+1) ln p und alle t in [0, 9/50] gilt  P'' >= p * margin  (margin > 0 ausgewiesen).
    Rueckgabe dict(ok, min_val (untere Schranke von P''/p minus eps_p/p), eps, boxes, fails)."""
    mut = mut or {}
    Rr = mut.get("R", R)
    Cc = mut.get("C", C)
    sigma0 = sig(2 * arb(Cc) * arb(p).log())
    if "sigma0" in mut:
        sigma0 = arb(mut["sigma0"])
    km = kappa_min(p, Cc)
    eps, info = eps_rest(p, km)
    if mut.get("eps_zero"):
        eps = arb(0)
    data = divisor_data(p, Rr)
    if grid is None:
        grid = s_grid()
    if sb is None:
        sb = s_boxes_lower(grid)
    # t in [0, 9/50]: obere Grenze als exakte Rationalzahl 9/50 -> Float-Obergrenze 0.18000000000000002 >= 0.18
    stack = []
    a, b = t_lo_float, t_hi_float
    n0 = first_split
    for k in range(n0):
        stack.append((a + (b - a) * k / n0, a + (b - a) * (k + 1) / n0))
    done = []
    fails = []
    pm1 = p + 1
    while stack:
        t0, t1 = stack.pop()
        A0l, A1l, Vl, Uu = t_box_lower(p, Cc, sigma0, data, t0, t1, mut)
        best = None
        for sl, nl, ml in sb:
            val = A0l + sl * A1l + pm1 * (Vl * nl + Uu * ml) - eps
            lo = val.lower()
            if best is None or lo < best:
                best = lo
        okbox = bool(best > margin * p)
        if okbox:
            done.append((t0, t1, best))
        else:
            if t1 - t0 < tol_width:
                fails.append((t0, t1, best))
            else:
                m = (t0 + t1) / 2
                stack.append((t0, m))
                stack.append((m, t1))
        if verbose and (len(done) + len(fails)) % 20 == 0:
            print("  boxes", len(done), "fails", len(fails), "stack", len(stack), flush=True)
    min_val = min((d[2] for d in done), default=None)
    arg = min(done, key=lambda d: d[2]) if done else None
    return {"ok": (not fails) and bool(done), "min_val": min_val, "min_over_p": (min_val / p if min_val is not None else None),
            "argmin_box": arg, "boxes": done, "eps": eps, "eps_info": info, "n_boxes": len(done), "fails": fails, "sigma0": sigma0, "kappa_min": km}
