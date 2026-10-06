# Certificates for Proposition 8.16 (uniqueness of the right zero of P_sigma)

These scripts accompany Section 8.4 and Appendix C of the paper *Fejér-Kernel Prime Indicators*
(arXiv:2506.18933). They carry out the interval-arithmetic computations behind

* Lemma 8.14 (strict convexity of `f(t) = P_sigma(p+t; kappa)` on `[0, 9/50]`),
* Lemma 8.15 (`f(0) < 0` and `f > 0` on `[9/50, 1)`),

for primes `p >= 5` and `kappa >= 3.25 (p+1) ln p`. **They are not a formal proof.** The soundness of the
libraries (Arb via python-flint, mpmath intervals) is assumed, and the status of every step is listed in
Table 1 of the paper (Appendix C.5); in particular two analytic auxiliary estimates for `p >= 23` are only
confirmed numerically (exact scan for `p <= 5000`, Arb for `p <= 101`) and not proved in full detail.

## Requirements

Python 3, `python-flint` 0.9.x, `mpmath` 1.3.x, `numpy`, `sympy` (only for prime lists). Windows and Linux.
Every script is self-contained within its directory and writes its log to `logs/` (and, where needed, to `out/`
or `data/`, which are kept as empty directories).

## Directories

| directory | what it certifies | main scripts (run in this order) | run time |
|---|---|---|---|
| `k2_inclusion/` | inclusion for the limit function `P_inf`: `P_inf(p+t) >= 0.0015 p` for `p >= 23`, `9/50 <= t < 1` (Appendix C.2) | `E7926_gate_...` (gate with mutants), `E7927_zertifikat_K2_...` (certificate `L_19(23,t) > 0.0015`), `E7928_lemma2_exakt_...` (guaranteed divisors against exact divisor sums, primes up to 2e6), `E7929_kette_direkt_...`, `E7930_klassen_mod6_...` | under 1 minute in total |
| `k2prime_inclusion/` | finite `kappa`: reduction for `p >= 23`, direct certificates for `p <= 19` (Appendix C.3) | `E7790_gate_zerlegung_...`, `E7791_reduktion_p23_...`, `E7792_rest_p_lt23_...` | 1 s, 6 min, 40 s |
| `k1_convexity/` | convexity `f'' > 0` on `[0, 9/50]`: `p <= 19` individually, `p >= 23` uniformly per residue class mod 6 (Appendix C.4) | gates `E7827_...`, `E7832_...`; certificates `E7828_...` (`p <= 19`), `E7833_...` (`p >= 23`); side conditions `E7834_...`; cross-check against direct summation at 52 cells `E7829_...` (mpmath, slow) | about 1.5 min, plus about 17 min for `E7829` |

The `E`-numbers are experiment numbers of the author's working environment and are kept for traceability.
`logs/` holds the logs of the runs quoted in the paper; a fresh run overwrites them.

All scripts were re-run from a fresh copy of this directory on 2026-10-06 (Windows, Python 3.13, python-flint 0.9.0, mpmath 1.3.0, numpy 2.4); every exit code was 0.

## What a green run means

A script prints `ok=True` (or `ERGEBNIS ... GRUEN`) if the finite cover of the parameter box succeeded and all
negative controls (deliberately wrong variants, "mutants") failed as they must. It does not mean that the
statement is proved without assumptions: see the status table of the paper.

## Provenance

Written by the author with the assistance of an AI system (Claude, Anthropic); reproduced by a second,
independently written implementation for the convexity certificate (`p <= 19` and the residue classes).
