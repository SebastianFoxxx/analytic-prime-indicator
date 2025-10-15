#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#
#   Numerical Verification and Plotting Script for the Framework of
#   Smooth, Analytic Analogues of Divisor Functions
#
#   This script provides a comprehensive suite for numerically verifying the
#   properties of the functions 𝒫(x), 𝒫_τ(x), and 𝒫_σ(x), and for
#   generating all plots presented in the accompanying paper.
#
#   Author: Sebastian Fuchs | sebastian.fuchs@hu-berlin.de | https://orcid.org/0009-0009-1237-4804
#   Date:   2025-10-15
#   Version: 1.2
#
# =============================================================================

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import mpmath
from sympy import sieve
from multiprocessing import Pool, cpu_count
from scipy.optimize import brentq
from tqdm import tqdm
import os

# =============================================================================
# CENTRAL CONFIGURATION
# =============================================================================
# All thresholds, ranges, and parameters can be adjusted here.

class Config:
    # --- Test Ranges for the foundational function 𝒫(x) ---
    ZERO_TEST_RANGE_HIGH_PRECISION = (1, 2000)
    ZERO_TEST_RANGE_FAST = (1, 100000)
    POSITIVITY_TEST_RANGE = (2.1, 20000.1)

    # --- Step Sizes ---
    POSITIVITY_STEP = 0.0005

    # --- Precision & Thresholds ---
    ZERO_PRECISION_THRESHOLD = 1e-9  # ε for zero checks and numeric floor for sign tests
    PRIME_EXCLUSION_TOL = 1e-6       # exclude |x - p| <= this band for odd primes in positivity scan
    MPMATH_PRECISION = 50
    
    # --- Parameters for the C^∞ Framework ---
    # Steepness parameter 'k' for the phi_kappa cutoff function.
    # A larger k makes the cutoff sharper, approximating a step function more closely.
    STEEPNESS_K = 1000.0

    # --- Resonance & RPF settings (host-level; Numba uses module constants below) ---
    RESONANCE_EPS = 1e-8  # guard for |x/i - round(x/i)|
    DENOM_THRESHOLD = 0.5 * math.sin(math.pi * RESONANCE_EPS)  # consistent with paper's epsilon-logic
    TAYLOR_MAX_DELTA_FACTOR = 0.25  # unused at call site but documented for clarity

    # --- RPF Consistency Test (resonant partial fractions) ---
    RPF_K = 2
    RPF_REL_TOL = 5e-4
    RPF_ABS_TOL = 1e-12
    RPF_K_CAP   = 4000
    RPF_TEST_SAMPLES = 200  # number of (x,i) sample points (resonant and generic)

    # --- Plotting Configuration ---
    PLOT_DPI = 300 # 300 for publication-quality, 100 for quick drafts
    PLOT_FILE_FORMAT = 'pdf' # alternatives: 'png', 'jpg' or 'pdf'
    
    # Figure sizes
    PLOT_OVERVIEW_FIGSIZE = (20, 5)
    PLOT_ZOOM_FIGSIZE = (10, 7)
    PLOT_DERIVATIVE_FIGSIZE = (12, 7)
    PLOT_PI_FIGSIZE = (12, 8)
    PLOT_TAU_SIGMA_FIGSIZE = (20, 5)
    PLOT_SIGMA_ZERO_FIGSIZE = (12, 8)
    PLOT_CUTOFF_FIGSIZE = (10, 6)

    # Plot ranges
    PLOT_OVERVIEW_RANGE = (2, 50)
    PLOT_P_TAU_RANGE = (2, 50)
    PLOT_P_SIGMA_RANGE = (0, 8) # Special range to show interesting zero behavior
    PLOT_CUTOFF_U_RANGE = (0, 2)
    PLOT_CUTOFF_K_VALUES = [5.0, 20.0, 100.0]

    # --- Prime Counting Function Plot Configuration ---
    PLOT_PI_RANGE = (0, 50)
    PLOT_PI_CONSTANT_C = 0.1  # constant threshold used in the illustrative sum
    ## H-variant (plot) parameters
    PLOT_H_ALPHA = 18.5   
    PLOT_H_GAMMA = 5.0   
    PLOT_H_EPS_PRIME_SAFETY = 100.0

    # --- Performance ---
    # Set to 0 to disable multiprocessing for easier debugging.
    NUM_PROCESSES = cpu_count()
    POSITIVITY_CHUNK_SIZE = 200_000  # number of x-points per task for the parallel positivity test
    ZERO_HP_CHUNK_SIZE = 100

# --- Numba-visible constants (must be defined at module level) ---
# Keep these numerically aligned with Config defaults.
RES_EPS = 1e-8
DENOM_THRESHOLD = 0.5 * math.sin(math.pi * RES_EPS)
TAYLOR_MAX_DELTA_FACTOR = 0.25
K_RPF_DEFAULT = 2

# =============================================================================
# CORE FUNCTION IMPLEMENTATIONS
# =============================================================================

# --- Foundational Function 𝒫(x) ---

@jit(nopython=True, fastmath=True)
def P_fast(x: float) -> float:
    """
    High-performance implementation of the foundational function 𝒫(x).
    Uses the closed Fejér identity F(x,i) = (sin(pi*x)/sin(pi*x/i))^2
    for O(√x) total work and improved numerical stability.
    """
    if x <= 1.0:
        return 0.0

    limit_i = math.ceil(math.sqrt(x))
    total_sum = 0.0
    sin_pi_x = math.sin(math.pi * x)

    for i in range(2, limit_i + 1):
        fejer_term = fejer_closed_form_term(x, i, sin_pi_x)
        total_sum += fejer_term

    return total_sum / x


def P_high_precision(x: int):
    """
    High-precision implementation of 𝒫(x) using the mpmath library for arbitrary-precision
    floating-point arithmetic. Essential for verifying exact zeros.
    """
    if x <= 1:
        return mpmath.mpf(0)

    mpmath.mp.dps = Config.MPMATH_PRECISION
    
    x_mp = mpmath.mpf(x)
    limit_i = mpmath.ceil(mpmath.sqrt(x_mp))
    total_sum = mpmath.mpf(0)

    for i in range(2, int(limit_i) + 1):
        i_mp = mpmath.mpf(i)
        inner_sum_val = i_mp
        for k in range(1, i):
            k_mp = mpmath.mpf(k)
            term = (i_mp - k_mp) * mpmath.cos(2 * mpmath.pi * x_mp * k_mp / i_mp)
            inner_sum_val += 2 * term
        total_sum += inner_sum_val

    return total_sum / x_mp

@jit(nopython=True)
def P_integer(n: int) -> float:
    """
    Fast exact evaluation of 𝒫(n) for integer n >= 2 using the divisor filter:
      𝒫(n) = (1/n) * sum_{2 <= d <= ceil(sqrt(n)), d|n} d^2
    This matches the definition with N(n) = ceil(sqrt(n)) and ensures 𝒫(2) > 0.
    """
    if n <= 1:
        return 0.0
    s = 0.0
    limit_i = int(math.ceil(math.sqrt(float(n))))
    for i in range(2, limit_i + 1):
        if n % i == 0:
            s += float(i * i)
    return s / float(n)


@jit(nopython=True)
def P_tau_integer(n: int, k: float) -> float:
    """
    Exact integer evaluation of 𝒫_τ(n; κ):
      𝒫_τ(n; κ) = (sum_{d|n, d>=2} φ_κ(d/(n+1)) * 1) - 1
    We enumerate divisors up to sqrt(n) and add the complementary factor n//d.
    This avoids any real-argument rounding issues at integer inputs.
    """
    if n <= 1:
        return -1.0
    s = 0.0
    limit_i = int(math.floor(math.sqrt(float(n))))
    for i in range(2, limit_i + 1):
        if n % i == 0:
            # Add divisor `i`.
            s += phi_kappa(i / (n + 1.0), k)
            # Add complementary divisor `j = n // i` if distinct.
            j = n // i
            if j != i and j >= 2:
                s += phi_kappa(j / (n + 1.0), k)
    # add the divisor d = n (>= 2) explicitly
    s += phi_kappa(n / (n + 1.0), k)
    return s - 1.0


# --- C^∞ Framework Components ---

@jit(nopython=True, fastmath=True)
def phi_kappa(u: float, k: float) -> float:
    """
    The C^∞ smooth transition cutoff function based on the hyperbolic tangent,
    as defined in the paper. It smoothly transitions from 1 to 0 around u=1.
    """
    return (1.0 - math.tanh(k * (u - 1.0))) / 2.0


@jit(nopython=True, fastmath=True)
def fejer_closed_form_term(x: float, i: int, sin_pi_x: float) -> float:
    """
    Compute F(x,i) using the closed Fejér identity with robust resonance guards:
        F(x,i) = (sin(pi*x) / sin(pi*x/i))^2
    Near resonances x/i ≈ Z, switch to a local even Taylor surrogate that matches
    the true expansion up to O((πδ)^2), with δ := x - i*round(x/i). This prevents
    catastrophic cancellation and overflow while preserving second-order accuracy.
    """
    # Exact divisor / removable singularity detection via delta
    m = int(round(x / i))
    delta = x - i * m
    if math.fabs(delta) <= 1e-16:
        # At x = i*m: limit equals i^2 exactly.
        return float(i * i)

    denom = math.sin(math.pi * x / i)

    # Resonance guard: denominator tiny OR |x/i - m| small -> Taylor surrogate
    if (math.fabs(denom) < DENOM_THRESHOLD) or (math.fabs(x / i - float(m)) < RES_EPS):
        # Constant-time local even Taylor surrogate around x = i*m
        # F(x,i) ≈ i^2 * (1 - α_i (π δ)^2)^2  with  α_i = (1/6) (1 - 1/i^2)
        alpha = (1.0 / 6.0) * (1.0 - 1.0 / (i * i))
        return i * i * (1.0 - alpha * (math.pi * delta) ** 2) ** 2

    # Non-resonant branch: safe quotient
    val = sin_pi_x / denom
    return val * val

@jit(nopython=True, fastmath=True)
def fejer_cosine_polynomial(x: float, i: int) -> float:
    """
    O(i) cosine-polynomial reference:
        F(x,i) = i + 2 * sum_{k=1}^{i-1} (i - k) * cos(2π x k / i).
    Used as the ground-truth reference in consistency tests.
    """
    s = float(i)
    two_pi_over_i = 2.0 * math.pi / float(i)
    for k in range(1, i):
        s += 2.0 * (i - k) * math.cos(two_pi_over_i * x * k)
    return s


@jit(nopython=True, fastmath=True)
def fejer_rpf_term(x: float, i: int, sin_pi_x: float, K: int) -> float:
    """
    Resonant partial-fraction (RPF) evaluator with symmetric truncation of (2K+1) poles.
    Exact identity:
        F(x,i) = (i^2 / π^2) * sin^2(π x) * Σ_{k∈Z} 1/(x - i k)^2.
    Here we truncate to the (2K+1) nearest poles around m = round(x/i).
    The tail magnitude satisfies Tail ≤ (2/π^2) * sin^2(π x) / K.
    """
    if K < 1:
        K = 1

    m = int(round(x / i))
    delta = x - i * m
    # Exact divisor: return the exact limit
    if math.fabs(delta) <= 1e-16:
        return float(i * i)

    s = 1.0 / (delta * delta)
    for r in range(1, K + 1):
        s += 1.0 / ((x - i * (m + r)) ** 2) + 1.0 / ((x - i * (m - r)) ** 2)
    return (i * i / (math.pi * math.pi)) * (sin_pi_x * sin_pi_x) * s

@jit(nopython=True, fastmath=True)
def P_tau_fast(x: float, k: float) -> float:
    """
    C^∞ smooth divisor-counting analogue 𝒫_τ(x) using the closed Fejér identity.
    Weighted by 1/i^2 and smoothed by phi_kappa(i/(x+1)).
    """
    if x <= 0.0:
        return -1.0  # consistent with definition (sum starts at i=2)

    limit_i = math.ceil(3 * x + 20) if x > 1.0 else 30
    total_sum = 0.0
    sin_pi_x = math.sin(math.pi * x)

    for i in range(2, limit_i + 1):
        fejer_term = fejer_closed_form_term(x, i, sin_pi_x)
        cutoff_val = phi_kappa(i / (x + 1.0), k)
        total_sum += cutoff_val * (fejer_term / (i * i))

    return total_sum - 1.0


@jit(nopython=True, fastmath=True)
def P_sigma_fast(x: float, k: float) -> float:
    """
    C^∞ smooth sum-of-divisors analogue 𝒫_σ(x) using the closed Fejér identity.
    Weighted by 1/i and smoothed by phi_kappa(i/(x+1)).
    """
    if x <= 0.0:
        return 0.0

    limit_i = math.ceil(3 * x + 20) if x > 1.0 else 30
    total_sum = 0.0
    sin_pi_x = math.sin(math.pi * x)

    for i in range(2, limit_i + 1):
        fejer_term = fejer_closed_form_term(x, i, sin_pi_x)
        cutoff_val = phi_kappa(i / (x + 1.0), k)
        total_sum += cutoff_val * (fejer_term / i)

    return total_sum - x

# =============================================================================
# NUMERICAL VERIFICATION SUITES
# =============================================================================

def check_integer_zero_high_precision(n: int) -> tuple[int, str] | None:
    """
    Worker function for multiprocessing. Checks the zero property of 𝒫(x) at integer n.
    """
    def is_prime_worker(num):
        if num <= 1: return False
        if num <= 3: return True
        if num % 2 == 0 or num % 3 == 0: return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0: return False
            i += 6
        return True

    val = P_high_precision(n)
    is_zero = abs(val) < Config.ZERO_PRECISION_THRESHOLD
    
    # Theorem 4.1 from the paper applies for x > 2
    if n <= 2:
        if n == 1 and not is_zero: return (n, "𝒫(1) is non-zero (should be zero by definition).")
        if n == 2 and is_zero: return (n, "𝒫(2) is zero (should be non-zero as 2 is prime but not an *odd* prime).")
    elif is_prime_worker(n): # Odd primes for n > 2
        if not is_zero: return (n, f"𝒫({n}) is non-zero (should be zero for odd prime).")
    else: # Composite numbers for n > 2
        if is_zero: return (n, f"𝒫({n}) is zero (should be non-zero for composite).")
            
    return None


def _positivity_worker(payload):
    """
    Worker for the positivity scan.
    payload = (points, primes, tol, zero_floor)
    Returns (processed_count, failures_list) where failures_list consists of (x, value) pairs.
    """
    points, primes, tol, zero_floor = payload
    local_failures = []
    for x in points:
        nearest = int(round(x))
        # Exclude band around odd primes to avoid near-zero sampling artifacts
        if (abs(x - nearest) <= tol) and (nearest in primes) and (nearest % 2 != 0):
            continue
        val = P_fast(x)
        # Only report true negatives beyond the numeric floor
        if val < -zero_floor:
            local_failures.append((x, val))
    return (len(points), local_failures)


def _zero_hp_worker(numbers):
    """
    Worker for the high-precision zero verification.
    Input: an iterable/list of integers.
    Returns: (processed_count, failures_list) where failures_list contains (n, msg).
    We set mpmath precision once per worker to avoid repeated resets.
    """
    try:
        mpmath.mp.dps = Config.MPMATH_PRECISION
    except Exception:
        pass

    local_failures = []
    processed = 0
    for n in numbers:
        res = check_integer_zero_high_precision(int(n))
        if res is not None:
            local_failures.append(res)
        processed += 1
    return processed, local_failures


class Verifier:
    """Encapsulates all verification tests for the foundational function 𝒫(x)."""

    def __init__(self, config: Config):
        self.config = config
        print("Initializing verifier...")
        max_range = max(
            self.config.ZERO_TEST_RANGE_HIGH_PRECISION[1],
            self.config.ZERO_TEST_RANGE_FAST[1], 
            int(self.config.POSITIVITY_TEST_RANGE[1]), 
            self.config.PLOT_PI_RANGE[1]
        )
        # Generate primes needed for tests and plots
        self.primes = set(sieve.primerange(1, max_range + 1))
        # Sanity check: prevent silent drift between host-level Config and numba-level constants
        def _approx_eq(a, b, tol=1e-15):
            return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

        if not _approx_eq(Config.RESONANCE_EPS, RES_EPS) \
           or not _approx_eq(Config.DENOM_THRESHOLD, DENOM_THRESHOLD) \
           or not _approx_eq(Config.TAYLOR_MAX_DELTA_FACTOR, TAYLOR_MAX_DELTA_FACTOR):
            print("WARNING: Resonance guard constants differ between Config and numba-level constants. "
                  "Align values to avoid unintended behavior.")


    def run_all_tests(self):
        """Runs the complete suite of verification tests for 𝒫(x)."""
        print("\n--- Starting Verification Suite for 𝒫(x) ---")
        self.run_zero_verification_high_precision()
        self.run_zero_verification_fast_vectorized()
        self.run_positivity_test()
        self.run_rpf_consistency_tests()
        self.run_companion_zero_validation()
        print("\n--- Verification Suite Finished ---\n")

    def run_zero_verification_high_precision(self):
        """
        Verifies zero properties of 𝒫(x) with high precision using multiprocessing.

        Smooth progress bar + good load balancing:
        - Interleave small and large n (dovetail order) to avoid long stalls at the end.
        - Dispatch small, fixed-size chunks to workers.
        - Each worker returns (processed_count, failures_chunk), so we can update tqdm
            accurately and continuously.
        """
        print(f"\n[TEST A1] Verifying Zeros of 𝒫(x) (High Precision) up to {self.config.ZERO_TEST_RANGE_HIGH_PRECISION[1]}...")
        n_min, n_max = self.config.ZERO_TEST_RANGE_HIGH_PRECISION
        all_nums = list(range(n_min, n_max + 1))

        # Create a dovetail/interleaved order: low, high, low+1, high-1, ...
        order = []
        lo, hi = 0, len(all_nums) - 1
        while lo <= hi:
            order.append(all_nums[lo]); lo += 1
            if lo <= hi:
                order.append(all_nums[hi]); hi -= 1

        chunk_size = int(getattr(self.config, "ZERO_HP_CHUNK_SIZE", 400))
        chunks = [order[i:i + chunk_size] for i in range(0, len(order), chunk_size)]

        failures = []
        start_time = time.time()

        if self.config.NUM_PROCESSES > 0:
            print(f"   INFO: Starting parallel verification on {self.config.NUM_PROCESSES} cores "
                f"(chunk_size={chunk_size}, tasks={len(chunks)}).")
            with tqdm(total=len(order), desc="   High-precision check", unit=" int", ncols=100) as progress_bar:
                with Pool(processes=self.config.NUM_PROCESSES, maxtasksperchild=200) as pool:
                    for processed, fails_chunk in pool.imap_unordered(_zero_hp_worker, chunks):
                        if fails_chunk:
                            failures.extend(fails_chunk)
                        progress_bar.update(processed)
                        progress_bar.set_postfix_str(f"Fails: {len(failures)}", refresh=True)
        else:
            print("   INFO: Starting sequential verification.")
            with tqdm(total=len(order), desc="   High-precision check", unit=" int", ncols=100) as progress_bar:
                for chunk in chunks:
                    # sequentially reuse the same logic as the worker for consistency
                    processed, fails_chunk = _zero_hp_worker(chunk)
                    if fails_chunk:
                        failures.extend(fails_chunk)
                    progress_bar.update(processed)
                    progress_bar.set_postfix_str(f"Fails: {len(failures)}", refresh=True)

        elapsed = time.time() - start_time
        print(f"   INFO: High-precision verification took {elapsed:.2f} seconds.")

        if not failures:
            print("   -> VERDICT: [PASS] All integers passed the high-precision check.")
        else:
            print(f"   -> VERDICT: [FAIL] {len(failures)} failures detected in high-precision check:")
            # Show first 5 failures for brevity
            for n, msg in sorted(failures)[:5]:
                print(f"        - At n={n}: {msg}")
            if len(failures) > 5:
                print(f"        ... and {len(failures) - 5} more.")


    def run_zero_verification_fast_vectorized(self):
        """Fast integer-domain zero check for 𝒫(n) using the divisor filter."""
        print(f"\n[TEST A2] Verifying Zeros of 𝒫(x) (Fast Integer) up to {self.config.ZERO_TEST_RANGE_FAST[1]}...")
        n_min, n_max = self.config.ZERO_TEST_RANGE_FAST
        start_time = time.time()

        numbers = np.arange(n_min, n_max + 1, dtype=np.int64)
        # Use the exact integer shortcut instead of evaluating P_fast on reals
        p_values = np.array([P_integer(int(n)) for n in numbers], dtype=np.float64)
        is_zero_mask = np.abs(p_values) < self.config.ZERO_PRECISION_THRESHOLD
        
        # According to Theorem 4.1, zeros for x>2 are odd primes.
        is_odd_prime_mask = np.array([int(n) in self.primes and int(n) % 2 != 0 for n in numbers])

        # False negatives: odd primes > 2 where P(n) is not zero.
        false_negatives = np.where(is_odd_prime_mask & (numbers > 2) & ~is_zero_mask)[0]
        # False positives: non (odd primes) > 2 where P(n) is zero.
        false_positives = np.where(~is_odd_prime_mask & (numbers > 2) & is_zero_mask)[0]
        
        failure_indices = np.concatenate([false_negatives, false_positives])
        elapsed = time.time() - start_time
        print(f"  INFO: Fast verification took {elapsed:.2f} seconds.")

        if len(failure_indices) == 0:
            print(f"   -> VERDICT: [PASS] All integers > 2 in range {self.config.ZERO_TEST_RANGE_FAST} passed the fast verification.")
        else:
            print(f"   -> VERDICT: [FAIL] {len(failure_indices)} failures detected in the fast verification:")
            for idx in failure_indices:
                n = int(numbers[idx])
                is_p = is_odd_prime_mask[idx]
                msg = f"𝒫({n}) is non-zero." if is_p else f"𝒫({n}) is zero."
                print(f"    - At n={n}: {msg}")

    def run_positivity_test(self):
        """
        Verifies that 𝒫(x) > 0 for non-prime x > 2.

        Numeric robustness:
        - exclude a small band around odd primes (|x - p| <= PRIME_EXCLUSION_TOL),
            since 𝒫(p) = 0 and floating grids can hit near-primes due to binary rounding;
        - only flag a failure if 𝒫(x) < -ZERO_PRECISION_THRESHOLD, i.e., a true negative
            beyond the numeric floor (tiny negatives are tolerated).

        Parallelization:
        - If NUM_PROCESSES > 0, split the grid into chunks and dispatch to workers.
        - Each worker returns (processed_count, failures_chunk), so the central tqdm bar
            can be updated accurately while preserving a single unified progress bar.
        """
        print(f"\n[TEST B] Verifying Positivity of 𝒫(x) in Range {self.config.POSITIVITY_TEST_RANGE}...")
        x_min, x_max = self.config.POSITIVITY_TEST_RANGE
        failures = []

        # Add a tiny jitter so we never sit exactly on integers due to cumulative rounding.
        rng = np.random.default_rng(12345)
        test_points = np.arange(x_min, x_max, self.config.POSITIVITY_STEP)
        test_points = test_points + rng.uniform(-1e-7, 1e-7, size=test_points.shape)

        tol = float(getattr(self.config, "PRIME_EXCLUSION_TOL", 1e-6))
        zero_floor = float(self.config.ZERO_PRECISION_THRESHOLD)
        use_parallel = int(getattr(self.config, "NUM_PROCESSES", 0)) > 0
        chunk_size = int(getattr(self.config, "POSITIVITY_CHUNK_SIZE", 100_000))

        if use_parallel:
            # Prepare chunked payloads for workers
            chunks = [test_points[i:i + chunk_size] for i in range(0, len(test_points), chunk_size)]
            # Convert to lists of floats to reduce pickling overhead
            payloads = [(chunk.tolist(), self.primes, tol, zero_floor) for chunk in chunks]

            processed_total = 0
            with tqdm(total=len(test_points), desc="   Positivity scan", unit=" x", ncols=100) as progress_bar:
                with Pool(processes=self.config.NUM_PROCESSES) as pool:
                    for processed, fails_chunk in pool.imap_unordered(_positivity_worker, payloads):
                        failures.extend(fails_chunk)
                        processed_total += processed
                        progress_bar.set_postfix_str(f"Fails: {len(failures)}", refresh=True)
                        progress_bar.update(processed)
        else:
            # Sequential fallback with a single progress bar
            with tqdm(total=len(test_points), desc="   Positivity scan", unit=" x", ncols=100) as progress_bar:
                for x in test_points:
                    nearest = int(round(x))
                    if (abs(x - nearest) <= tol) and (nearest in self.primes) and (nearest % 2 != 0):
                        progress_bar.update(1)
                        continue
                    val = P_fast(x)
                    if val < -zero_floor:
                        failures.append((x, val))
                    progress_bar.set_postfix_str(f"Fails: {len(failures)}", refresh=True)
                    progress_bar.update(1)

        # Report
        if not failures:
            print("   -> VERDICT: [PASS] Function is positive at all tested non-prime points > 2 (within numeric tolerances).")
        else:
            print(f"   -> VERDICT: [FAIL] {len(failures)} points found where P(x) is negative beyond tolerance:")
            for x_fail, v_fail in failures[:5]:
                print(f"        - P(x) = {v_fail:.3e} at x = {x_fail:.6f}")
            if len(failures) > 5:
                print(f"        ... and {len(failures) - 5} more.")



    def run_rpf_consistency_tests(self):
        """
        [TEST C] Consistency test between:
        (1) Closed-form with resonance guards (fejer_closed_form_term),
        (2) RPF truncated evaluator with adaptive K (fejer_rpf_term),
        (3) Cosine-polynomial reference (fejer_cosine_polynomial).
        For each (x,i), choose K so that the tail bound
            Tail ≤ (2/π^2) * sin^2(π x) / K
        is below max(abs_tol, rel_tol * max(1, |F_ref|)).
        Diagnostics report a scaled error consistent with the mismatch criterion,
        and a separate absolute error when |F_ref| is near zero.
        """
        print("\n[TEST C] Verifying RPF/Closed-form/Cosine consistency ...")
        rel_tol = float(self.config.RPF_REL_TOL)
        abs_tol = float(self.config.RPF_ABS_TOL)
        K_cap   = int(self.config.RPF_K_CAP)

        # Sample points: generic, near-integer (sin(πx)≈0), and near-resonant x≈im±ε i
        xs = list(np.linspace(2.1, 50.0, int(self.config.RPF_TEST_SAMPLES)))
        xs += [n + 1e-6 for n in range(3, 51)]
        for i in range(2, 60, 3):
            for m in range(1, 6):
                base = i * m
                xs.append(base + 1e-6 * i)
                xs.append(base - 1e-6 * i)

        failures_closed = 0
        failures_rpf = 0
        kcap_saturated = 0
        tested = 0        
        worst_scaled = (0.0, None)   # (scaled_err, (x,i,K_use,f_rp,f_cos))
        worst_abs    = (0.0, None)   # (abs_err,    (x,i,K_use,f_rp,f_cos))
        k_sum = 0
        k_max = 0

        # Cosine reference cost is O(i); keep i moderate.
        for x in xs:
            sin_pi_x = math.sin(math.pi * x)
            i_max = min(120, max(10, int(math.ceil(math.sqrt(max(2.0, x))) + 50)))
            for i in range(2, i_max + 1, 3):
                f_cos = fejer_cosine_polynomial(x, i)
                f_cl  = fejer_closed_form_term(x, i, sin_pi_x)

                # Closed-form vs reference
                ref_mag = max(1.0, abs(f_cos))
                cerr = abs(f_cl - f_cos)
                if (cerr / ref_mag > rel_tol) and (cerr > abs_tol):
                    failures_closed += 1

                # Adaptive K selection for RPF using the tail bound
                tol_abs = max(abs_tol, rel_tol * ref_mag)
                s2 = sin_pi_x * sin_pi_x
                # Compute required K from tail bound and enforce a minimum from Config
                K_needed = 1
                if tol_abs > 0.0:
                    K_needed = int(math.ceil((2.0 / (math.pi * math.pi)) * s2 / tol_abs))
                    if K_needed < 1:
                        K_needed = 1
                K_min = max(1, int(self.config.RPF_K))
                K_use = max(K_min, K_needed)
                if K_use > K_cap:
                    K_use = K_cap

                f_rp  = fejer_rpf_term(x, i, sin_pi_x, K_use)
                rerr  = abs(f_rp - f_cos)
                # Soft diagnostic: K reached cap but theoretical tail bound still above tol_abs
                if K_use == K_cap:
                    tail_bound = (2.0 / (math.pi * math.pi)) * s2 / K_use
                    if tail_bound > tol_abs:
                        kcap_saturated += 1

                # Mismatch decision consistent with ref_mag scaling
                if (rerr / ref_mag > rel_tol) and (rerr > abs_tol):
                    failures_rpf += 1
                else:
                    scaled_err = rerr / ref_mag
                    if abs(f_cos) < 10.0 * abs_tol:
                        # Reference ~ 0: track absolute error separately
                        if rerr > worst_abs[0]:
                            worst_abs = (rerr, (x, i, K_use, f_rp, f_cos))
                    else:
                        if scaled_err > worst_scaled[0]:
                            worst_scaled = (scaled_err, (x, i, K_use, f_rp, f_cos))

                tested += 1
                k_sum += K_use
                if K_use > k_max:
                    k_max = K_use

        print(f"  INFO: RPF test pairs evaluated: {tested}")
        if tested > 0:
            print(f"  INFO: Adaptive K stats — average: {k_sum/max(1,tested):.1f}, max: {k_max}")

        if failures_closed == 0 and failures_rpf == 0:
            print("   -> VERDICT: [PASS] Closed-form and RPF are consistent with the cosine reference within set tolerances.")
        else:
            print(f"   -> VERDICT: [FAIL] Inconsistencies detected (tolerances rel={rel_tol:g}, abs={abs_tol:g}):")
            print(f"           Closed-form mismatches : {failures_closed}")
            print(f"           RPF mismatches        : {failures_rpf}")
        if kcap_saturated > 0:
            print(f"  INFO: K_cap saturation events: {kcap_saturated} (theoretical tail bound > abs_tol).")

        # Diagnostics: report scaled error (consistent with mismatch rule) and abs error near zero reference
        if worst_scaled[1] is not None:
            xw, iw, Kw, frp, fcos = worst_scaled[1]
            margin_rel = float(self.config.RPF_REL_TOL) - worst_scaled[0]
            print(f"  INFO: Largest observed scaled deviation among matches: {worst_scaled[0]:.3e} "
                  f"(margin to rel_tol: {margin_rel:.3e}) at (x={xw:.6f}, i={iw}, K={Kw})")
        if worst_abs[1] is not None:
            xw, iw, Kw, frp, fcos = worst_abs[1]
            margin_abs = float(self.config.RPF_ABS_TOL) - worst_abs[0]
            note = "" if margin_abs >= 0 else " — abs_tol exceeded alone; no mismatch because the relative criterion was not exceeded"
            print(f"  INFO: Largest absolute deviation with near-zero reference: {worst_abs[0]:.3e} "
                  f"(margin to abs_tol: {margin_abs:.3e}) at (x={xw:.6f}, i={iw}, K={Kw}){note}")

    def run_companion_zero_validation(self):
        """
        [TEST E] Compact semantic validation for companion zero conjectures (6.3 & 7.3).

        Design goals:
        - Test against the first 100 odd primes.
        - Provide a consolidated summary of PASS/FAIL results.
        - Display a progress bar during validation.
        """
        
        print("\n[TEST E] Semantically Validating Companion Zero Conjectures...")

        # --- Configuration (tunable but conservative) ---
        # Generate the first 100 odd primes for testing (from 3 to 547)
        all_primes = list(sieve.primerange(1, 2000))
        primes_to_test = all_primes[1:]
        
        kappa_values   = sorted([5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0])

        precision_threshold = 1e-9          # numeric floor for distances
        floor   = 10.0 * precision_threshold  # "trustworthy" lower bound
        ceiling = 1e-2                        # "unsaturated" upper bound for mid-range checks
        ratio_cap_tau = 150.0               # P_tau symmetry cap (scale = max(dl/dr, dr/dl))
        min_right_sigma = 1e-3              # P_sigma right distance must stay visibly > 0 at high kappa

        # --- Minimal bracketing helper (adaptive but compact) ---
        def _bracket_and_solve_zero(f, center, direction, initial_span=0.25, max_span=5.0, growth=2.0, max_tries=16):
            """
            Expand from x=center in 'direction' (+1 right, -1 left) until a sign change is found; solve with brentq.
            Returns root or None.
            """
            fa = f(center)
            span = initial_span
            tries = 0
            while tries < max_tries and span <= max_span:
                a = center - span if direction < 0 else center
                b = center if direction < 0 else center + span
                try:
                    fb = f(b) if direction > 0 else f(a)
                except Exception:
                    return None
                if np.sign(fa) != np.sign(fb):
                    lo, hi = (a, center) if direction < 0 else (center, b)
                    try:
                        return brentq(f, lo, hi)
                    except Exception:
                        return None
                span *= growth
                tries += 1
            return None

        def find_zeros(p, k, func):
            """Return (left_zero, right_zero) or (None, None) if bracketing failed."""
            f = lambda x: func(x, k)
            zl = _bracket_and_solve_zero(f, p, direction=-1)
            zr = _bracket_and_solve_zero(f, p, direction=+1)
            return zl, zr

        def eventual_decrease(seq_by_k, ks, min_decreases=1, allowed_violations=1):
            """
            Check 'eventual decrease' on values mapped by kappa:
            - Count decreases when the previous value is above 'floor'.
            - Require at least 'min_decreases' true decreases.
            - Allow up to 'allowed_violations' non-decreasing steps.
            """
            prev = None
            decreases = 0
            violations = 0
            eligible = 0
            for k in ks:
                v = seq_by_k[k]
                if prev is not None and prev > floor:
                    eligible += 1
                    if v < prev:
                        decreases += 1
                    else:
                        violations += 1
                prev = v
            if eligible == 0:
                return True  # nothing meaningful to test; do not fail on noise
            return (decreases >= min_decreases) and (violations <= allowed_violations)

        def pick_mid_k_for_symmetry(valid_dists, ks):
            """
            Pick a single 'mid-high' kappa where both sides are in [floor, ceiling].
            Strategy: scan from high to low and return the first k that fits the band.
            Returns k or None.
            """
            for k in reversed(ks):
                dl, dr = valid_dists[k]
                if (dl >= floor and dr >= floor and dl <= ceiling and dr <= ceiling):
                    return k
            return None

        # --- 1) P_tau: symmetric convergence ---
        print("\n 1. Testing P_tau (Conjecture 6.3: Symmetric Convergence) ...")
        tau_failures = []
        
        # Progress bar for P_tau tests        
        with tqdm(total=len(primes_to_test), desc="   Phase 1/2: P_tau checks", unit=" prime", ncols=100) as progress_bar:
            for p in primes_to_test:
                dists = {k: find_zeros(p, k, P_tau_fast) for k in kappa_values}
                valid = {k: (abs(p - v[0]), abs(v[1] - p))
                        for k, v in dists.items() if v[0] is not None and v[1] is not None}

                reasons = []
                if len(valid) < 2:
                    reasons.append("Not enough valid zero pairs")
                else:
                    ks = sorted(valid.keys())
                    left_vals  = {k: valid[k][0] for k in ks}
                    right_vals = {k: valid[k][1] for k in ks}

                    if not eventual_decrease(left_vals, ks): reasons.append("Left convergence not eventual")
                    if not eventual_decrease(right_vals, ks): reasons.append("Right convergence not eventual")

                    k_star = pick_mid_k_for_symmetry(valid, ks)
                    if k_star is not None:
                        dl, dr = valid[k_star]
                        scale = max(dl / dr, dr / dl) if min(dl, dr) > precision_threshold else float('inf')
                        if scale > ratio_cap_tau:
                            reasons.append(f"Symmetry imbalanced at k={k_star} (scale={scale:.2f})")
                
                if reasons:
                    tau_failures.append((p, ", ".join(sorted(set(reasons)))))
                
                progress_bar.set_postfix_str(f"Fails: {len(tau_failures)}", refresh=True)
                progress_bar.update(1)

        print()

        # Final summary for P_tau
        if not tau_failures:
            print("   -> OVERALL P_tau VERDICT: [PASS]")
            print("      All tested primes passed the symmetric convergence checks.")
        else:
            print(f"   -> OVERALL P_tau VERDICT: [FAIL] ({len(tau_failures)}/{len(primes_to_test)} primes failed)")
            for p, reason_str in tau_failures:
                print(f"        - p={p:<4}: {reason_str}")
        print()


        # --- 2) P_sigma: asymmetric convergence ---
        print(" 2. Testing P_sigma (Conjecture 7.3: Asymmetric Convergence) ...")
        sigma_failures = []

        # Progress bar for P_sigma tests
        with tqdm(total=len(primes_to_test), desc="   Phase 2/2: P_sigma checks", unit=" prime", ncols=100) as progress_bar:
            for p in primes_to_test:
                dists = {k: find_zeros(p, k, P_sigma_fast) for k in kappa_values}
                valid = {k: (abs(p - v[0]), abs(v[1] - p))
                        for k, v in dists.items()
                        if v[0] is not None and v[1] is not None and abs(v[1] - p) > precision_threshold}

                reasons = []
                if len(valid) < 2:
                    reasons.append("Not enough reliable zero pairs found (solver failed)")
                else:
                    ks = sorted(valid.keys())
                    left_vals  = {k: valid[k][0] for k in ks}
                    right_vals = {k: valid[k][1] for k in ks}

                    if not eventual_decrease(left_vals, ks): reasons.append("Left convergence not eventual")
                    
                    last_k = ks[-1]
                    if not (right_vals[last_k] > min_right_sigma):
                        reasons.append(f"Right zero not visibly away at high kappa (dist={right_vals[last_k]:.2e})")

                if reasons:
                    sigma_failures.append((p, ", ".join(sorted(set(reasons)))))
                
                progress_bar.set_postfix_str(f"Fails: {len(sigma_failures)}", refresh=True)
                progress_bar.update(1)
        
        print()
            
        # Final summary for P_sigma
        if not sigma_failures:
            print("   -> OVERALL P_sigma VERDICT: [PASS]  ")
            print("      All tested primes passed the asymmetric convergence checks.")
        else:
            print(f"   -> OVERALL P_sigma VERDICT: [FAIL] ({len(sigma_failures)}/{len(primes_to_test)} primes failed)")
            for p, reason_str in sigma_failures:
                print(f"        - p={p:<4}: {reason_str}")
        print()



# =============================================================================
# PRIME-COUNTING: NON-ACCUMULATIVE H-VARIANT (ELEMENTARY, CASE-FREE)
# =============================================================================

@jit(nopython=True, fastmath=True)
def kappa_schedule(n: int, alpha: float) -> float:
    """
    Linear schedule for the smooth cutoff steepness: κ(n) = alpha * (n + 1).
    This reliably separates primes (g ~ 0) from composites (g ≳ 1) for moderate alpha.
    """
    return alpha * (n + 1.0)

@jit(nopython=True, fastmath=True)
def eps_poly(n: int, gamma: float) -> float:
    """
    Polynomial regularizer: ε(n) = (n + 1)^(-gamma), gamma > 1.
    Controls the global sup error by sum_{n>=2} ε(n) = ζ(gamma) - 1.
    """
    return 1.0 / ((n + 1.0) ** gamma)

@jit(nopython=True, fastmath=True)
def eps_subexp(n: int, c: float) -> float:
    """
    Subexponential regularizer: ε(n) = exp(-c * log^2(n + 1)).
    Decays faster than any power; allows extremely small global sup bounds.
    """
    ln = math.log(n + 1.0)
    return math.exp(-c * ln * ln)

@jit(nopython=True, fastmath=True)
def t_indicator(n: int, alpha: float, mode: int, param: float) -> float:
    """
    Case-free prime indicator term:
        t_n = ε(n) / (g(n) + ε(n)),
    where g(n) = |P_tau_integer(n; κ(n))| and κ(n) = alpha * (n + 1).
    mode == 0: ε(n) = (n + 1)^(-param)  with param=gamma
    mode == 1: ε(n) = exp(-param * log^2(n + 1)) with param=c
    """
    kappa = kappa_schedule(n, alpha)
    g = abs(P_tau_integer(n, kappa))
    if mode == 0:
        eps = eps_poly(n, param)
    else:
        eps = eps_subexp(n, param)
    return eps / (g + eps)

def run_prime_counting_H_benchmark(
    primes: set,
    config,
    x_max: int = 5000,
    alpha_grid = (16, 17, 18, 19, 20, 21),
    gamma_grid = (3.5, 4.0, 5.0, 6.0, 7.0),
    c_grid = (0.3, 0.5, 0.8, 1.0),
    C_baseline: float = 1e-3,
    report_fair_baseline: bool = True,
) -> None:
    """
    Benchmark for the H-variant vs. constant-C baselines.

    H-variant (summable composite-error):
      - kappa(n)   = alpha * (n + 1)
      - epsilon(n) = (n + 1)^(-gamma)
      - term_n     = epsilon(n) / (|P_tau(n; kappa(n))| + epsilon(n))
      - pi_hat(x)  = sum_{n<=x} term_n

    Baseline A (didactic, drift-prone):
      - kappa      = constant (config.STEEPNESS_K)
      - epsilon    = constant C_baseline

    Baseline B (fair, optional):
      - kappa(n)   = alpha_best * (n + 1)  [same scaling as H]
      - epsilon    = constant C_baseline

    Logs:
      - Correctly interpreted composite error bound.
      - Tanh(alpha) saturation diagnostics.
      - Sup/mean/quantile errors, linear drift slope for baselines.
      - Rounding-match counts.
    """
    import numpy as np
    import mpmath as mp
    from math import tanh

    print("\n[TEST D] Prime-counting H-variant (summable composite-error) vs constant-C baseline")
    print(f"  INFO: Evaluation range x ∈ [2, {x_max}] on integers only.")
    print("  INFO: First Numba call may trigger compilation (one-time overhead).")

    # ---- Prepare true π(x) on integers --------------------------------------------------------
    ints = np.arange(2, x_max + 1, dtype=np.int64)
    is_prime = np.array([int(n) in primes for n in ints], dtype=bool)
    pi_true = np.cumsum(is_prime, dtype=np.int64)

    # Utility to compute rounded-match count
    def count_rounded_matches(pi_hat: np.ndarray) -> int:
        return int(np.sum((np.floor(pi_hat + 0.5)).astype(np.int64) == pi_true))

    # Utility to summarize errors
    def summarize_errors(pi_hat: np.ndarray, name: str, with_drift: bool = True):
        err = pi_hat - pi_true.astype(float)
        sup_err = float(np.max(np.abs(err)))
        mae = float(np.mean(np.abs(err)))
        p95 = float(np.quantile(np.abs(err), 0.95))
        matches = count_rounded_matches(pi_hat)
        print(f"\n  {name}")
        print(f"    Sup error            = {sup_err:.10f}")
        print(f"    Mean abs. error      = {mae:.10f}")
        print(f"    95% quantile |err|   = {p95:.10f}")
        if with_drift:
            x = ints.astype(float)
            a, _ = np.polyfit(x, err, 1)
            print(f"    Linear drift slope   = {a:.10e} (error ≈ slope·x + const)")
        print(f"    Rounding matches     = {matches}/{len(ints)}")
        return sup_err, mae, p95, matches

    # ---- Sweep H-variant over (alpha, gamma) --------------------------------------------------
    best = {
        "alpha": None, "gamma": None,
        "sup": float("inf"), "mae": float("inf"),
        "pi_hat": None, "g_scaled": None
    }
    mp.mp.dps = 50

    for alpha in alpha_grid:
        tanh_alpha = tanh(alpha)
        print(f"  INFO: α={alpha:>6}: tanh(α)={tanh_alpha:.16f}, 1 - tanh(α)≈{1.0 - tanh_alpha:.1e}")
        kappa_vec = alpha * (ints.astype(float) + 1.0)
        p_tau_scaled = np.array([P_tau_integer(int(n), float(k)) for n, k in zip(ints, kappa_vec)], dtype=np.float64)
        g_scaled = np.abs(p_tau_scaled)

        for gamma in gamma_grid:
            eps = (ints.astype(float) + 1.0) ** (-float(gamma))
            terms = eps / (g_scaled + eps)
            pi_hat = np.cumsum(terms)

            err = pi_hat - pi_true.astype(float)
            sup_err = float(np.max(np.abs(err)))
            mae = float(np.mean(np.abs(err)))
            is_better = (sup_err < best["sup"]) or (sup_err == best["sup"] and mae < best["mae"])
            if is_better:
                best.update(alpha=alpha, gamma=gamma, sup=sup_err, mae=mae, pi_hat=pi_hat.copy(), g_scaled=g_scaled.copy())

    # ---- Report H-variant result with correct bound interpretation ----------------------------
    print("\n  RESULT (H-variant with summable composite-error):")
    print("    Best schedule        = ε(n)=(n+1)^(-gamma)  [polynomial]")
    print(f"    Best (alpha, gamma)  = ({best['alpha']}, {best['gamma']}) on x ≤ {x_max}")
    gamma_best = float(best["gamma"])
    z = float(mp.zeta(gamma_best))
    composite_error_bound = z - 1.0 - 2.0**(-gamma_best) - 3.0**(-gamma_best)
    sup_H, mae_H, p95_H, matches_H = summarize_errors(best["pi_hat"], "H-variant (summable composite-error)", with_drift=False)
    print(f"    Composite error bound = {composite_error_bound:.10f} (upper limit for non-accumulative part)")

    # ---- Baseline A: constant C, non-scaled kappa (didactic; shows drift) ---------------------
    print("\n  BASELINE A (constant-C, non-scaled κ) for comparison:")
    kappa_const = float(config.STEEPNESS_K)
    p_tau_const = np.array([P_tau_integer(int(n), kappa_const) for n in ints], dtype=np.float64)
    g_const = np.abs(p_tau_const)
    terms_baseA = float(C_baseline) / (g_const + float(C_baseline))
    pi_hat_A = np.cumsum(terms_baseA)
    sup_A, mae_A, p95_A, matches_A = summarize_errors(pi_hat_A, "Baseline A")

    # ---- Baseline B: constant C, scaled kappa (fair baseline, reusing best g_scaled) ----------
    sup_B = mae_B = p95_B = matches_B = None
    if report_fair_baseline and best["g_scaled"] is not None:
        print("\n  BASELINE B (constant-C, scaled κ=alpha_best·(n+1)) for comparison:")
        g_scaled_best = best["g_scaled"]
        terms_baseB = float(C_baseline) / (g_scaled_best + float(C_baseline))
        pi_hat_B = np.cumsum(terms_baseB)
        sup_B, mae_B, p95_B, matches_B = summarize_errors(pi_hat_B, "Baseline B")

    # ---- Final Comparison Summary -----------------------------------------------------------
    print("\n  COMPARISON SUMMARY:")
    print(f"    H-variant:  sup={sup_H:.10f},  mae={mae_H:.10f},  round_all={matches_H==len(ints)}")
    if report_fair_baseline:
        print(f"    Baseline A: sup={sup_A:.10f},  mae={mae_A:.10f},  round_all={matches_A==len(ints)}")
        print(f"    Baseline B: sup={sup_B:.10f},  mae={mae_B:.10f},  round_all={matches_B==len(ints)}")
    else:
        print(f"    Baseline A: sup={sup_A:.10f},  mae={mae_A:.10f},  round_all={matches_A==len(ints)}")

    print("\n[TEST D] Finished.")

# =============================================================================
# PLOTTING SUITE
# =============================================================================

class Plotter:
    """Generates publication-quality plots for all relevant functions."""
    
    def __init__(self, config: Config, primes: set):
        self.config = config
        self.primes = primes
        plt.style.use('seaborn-v0_8-whitegrid')

    def run_all_plots(self):
        """Generates the complete suite of plots for the paper."""
        print("\n--- Generating Plots ---")
        self.generate_overview_plot()
        self.generate_zoom_plot()
        self.generate_derivative_plot()
        self.generate_prime_counting_plot()
        self.generate_cutoff_plot()
        self.generate_P_tau_plot()
        self.generate_P_sigma_plot()
        self.generate_companion_zeros_plot()
        self.generate_companion_zeros_sigma_plot()
        print("--- Plot Generation Finished ---\n")

    def generate_overview_plot(self):
        """Plot of the foundational function 𝒫(x) with odd prime markers."""
        x_min, x_max = self.config.PLOT_OVERVIEW_RANGE
        x = np.linspace(x_min, x_max, 8000)
        y = np.vectorize(P_fast)(x)
        
        fig, ax = plt.subplots(figsize=self.config.PLOT_OVERVIEW_FIGSIZE)
        ax.plot(x, y, lw=1.0, color='royalblue')
        ax.axhline(0, color='black', lw=0.7)
        primes_in_range = sorted([p for p in self.primes if x_min < p < x_max and p % 2 != 0])
        for p in primes_in_range:
            ax.axvline(p, color='green', linestyle=':', lw=1.5, alpha=0.7)
        ax.set_title(r"Overview of the Foundational Function $\mathcal{P}(x)$", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}(x)$", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(primes_in_range)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12, rotation=90)
        filename = f"plot_overview.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  Generated foundational function overview plot: '{filename}'.")
        plt.close(fig)

    def generate_zoom_plot(self):
        """Zoom-in on a representative zero of 𝒫(x) at an odd prime."""
        x_prime = 13
        x = np.linspace(x_prime - 0.1, x_prime + 0.1, 2000)
        y = np.vectorize(P_fast)(x)
        fig, ax = plt.subplots(figsize=self.config.PLOT_ZOOM_FIGSIZE)
        ax.plot(x, y, '.-', lw=1.0, ms=4, color='darkorange')
        ax.axhline(0, color='black', lw=0.7)
        ax.axvline(x_prime, color='red', alpha=0.5, linestyle='--', lw=2.0, label=f'$x = {x_prime}$ (odd prime)')
        ax.set_title(r"Detailed View of the Zero at $x=13$", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}(x)$", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend()
        filename = f"plot_zoom_{x_prime}.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated plot to '{filename}'.")
        plt.close(fig)

    def generate_derivative_plot(self):
        """Plots the numerical second derivative of 𝒫(x), showing discontinuities."""
        x_min, x_max = 3.5, 16.5
        x = np.linspace(x_min, x_max, 8000)
        d = 1e-5
        y2 = np.vectorize(
            lambda val: (P_fast(val + d) - 2 * P_fast(val) + P_fast(val - d)) / d**2
        )(x)
        fig, ax = plt.subplots(figsize=self.config.PLOT_DERIVATIVE_FIGSIZE)
        ax.plot(x, y2, lw=1.5, color='darkviolet')
        ax.set_title(r"Numerical Second Derivative $\mathcal{P}''(x)$", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}''(x)$ (approx.)", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', labelsize=12)
        squares_in_range = [k**2 for k in [2, 3, 4] if x_min < k**2 < x_max]
        for i, sq in enumerate(squares_in_range):
            label = r'Discontinuity at $x=m^2$' if i == 0 else ""
            ax.axvline(sq, color='red', alpha=0.5, linestyle='--', lw=2.0, label=label)
        ax.legend()
        filename = f"plot_second_derivative.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated second derivative plot to '{filename}'.")
        plt.close(fig)
    
    def generate_prime_counting_plot(self):
        """
        Comparison plot: constant-C baseline (drift) vs non-accumulative H-variant,
        both overlaid with the true staircase π(x), on x in [0, PLOT_PI_RANGE[1]].
        """
        x_min, x_max = self.config.PLOT_PI_RANGE
        C = float(self.config.PLOT_PI_CONSTANT_C)          # constant-threshold baseline
        k_const = float(self.config.STEEPNESS_K)           # fixed κ for baseline
        alpha = float(self.config.PLOT_H_ALPHA)            # H-variant: κ(n) = α(n+1)
        gamma = float(self.config.PLOT_H_GAMMA)            # H-variant: ε(n) = (n+1)^(-γ), γ>1

        # ensure epsilon on the plotted range dwarfs the prime residual g ≈ e^{-2α}
        S = float(getattr(self.config, "PLOT_H_EPS_PRIME_SAFETY", 100.0))
        gamma_cfg = float(gamma)
        gamma_cap = (2.0*alpha - math.log(S)) / math.log(x_max + 1.0)  # upper bound for γ
        if gamma_cfg > gamma_cap:
            print(f"  WARN: H-γ={gamma_cfg} too large for (α={alpha}, x_max={x_max}); capping to {gamma_cap:.3f} "
                f"so that ε(x_max) ≥ {S}·e^(-2α) and primes are not undercounted.")
            gamma = gamma_cap
        else:
            gamma = gamma_cfg

        # integer grid
        n_grid = np.arange(2, x_max + 1, dtype=np.int64)

        # --- True π(x) staircase on 0..x_max
        plot_x = np.arange(0, x_max + 1, dtype=np.int64)
        true_pi = np.zeros_like(plot_x, dtype=int)
        cnt = 0
        for n in range(2, x_max + 1):
            if n in self.primes:
                cnt += 1
            true_pi[n] = cnt

        # --- Baseline (constant C, constant κ)
        p_tau_const = np.array([P_tau_integer(int(n), k_const) for n in n_grid], dtype=np.float64)
        g_const = np.abs(p_tau_const)
        terms_baseline = 1.0 - (g_const / (g_const + C))     # = ε/(g+ε) with ε=C
        terms_baseline = np.clip(terms_baseline, 0.0, 1.0)
        pi_baseline = np.cumsum(terms_baseline)

        # Lift to 0..x_max (π(0)=π(1)=0)
        y_baseline = np.concatenate(([0, 0], pi_baseline))

        # --- H-variant (non-accumulative): κ(n)=α(n+1), ε(n)=(n+1)^(-γ)
        kappa_vec = alpha * (n_grid.astype(float) + 1.0)
        p_tau_scaled = np.array([P_tau_integer(int(n), float(k)) for n, k in zip(n_grid, kappa_vec)],
                                dtype=np.float64)
        g_scaled = np.abs(p_tau_scaled)
        eps_vec = (n_grid.astype(float) + 1.0) ** (-gamma)
        terms_H = 1.0 - (g_scaled / (g_scaled + eps_vec))
        terms_H = np.clip(terms_H, 0.0, 1.0)
        pi_H = np.cumsum(terms_H)

        y_H = np.concatenate(([0, 0], pi_H))

        # --- Plot
        fig, ax = plt.subplots(figsize=self.config.PLOT_PI_FIGSIZE)

        # True staircase
        ax.step(plot_x, true_pi, where='post', lw=1.2, color='black', alpha=0.75, label=r'True $\pi(x)$')

        # Baseline (drift is visible due to the choice of C)
        ax.step(plot_x, y_baseline, where='post', lw=1.2, color='tab:orange',
                label=rf'Baseline (constant $C={C}$, constant $\kappa$)')

        # H-variant (closely tracks ground truth)
        ax.step(plot_x, y_H, where='post', lw=1.2, color='royalblue',
                label=rf'$H$-variant ($\kappa(n)=\alpha(n+1)$, $\alpha={alpha:g}$, $\varepsilon(n)=(n+1)^{{-\gamma}}$, $\gamma={gamma:g}$)')

        # Axes/labels
        ax.set_title(r"Prime counting via $\mathcal{P}_{\tau}$: constant-$C$ baseline vs non-accumulative $H$-variant",
                    fontsize=16)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"Count", fontsize=14)
        ax.set_xlim(x_min, x_max)

        # y-range: integer ticks up to the max among curves
        y_max = int(max(true_pi.max(), y_baseline.max(), y_H.max())) + 1
        ax.set_ylim(bottom=-0.5, top=y_max)
        ax.set_yticks(np.arange(0, y_max + 1, 1))
        ax.set_xticks(np.arange(x_min, x_max + 1, 5))
        ax.tick_params(axis='both', labelsize=12)

        ax.legend(fontsize=11, loc='upper left', ncol=1)

        filename = f"plot_prime_counting.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated comparative prime-counting plot to '{filename}'.")
        plt.close(fig)

    def generate_cutoff_plot(self):
        """Visualizes the C^∞ smooth cutoff function phi_kappa for different k."""
        u_min, u_max = self.config.PLOT_CUTOFF_U_RANGE
        k_values = self.config.PLOT_CUTOFF_K_VALUES
        u = np.linspace(u_min, u_max, 1000)
        
        fig, ax = plt.subplots(figsize=self.config.PLOT_CUTOFF_FIGSIZE)
        for k in k_values:
            y = np.vectorize(phi_kappa)(u, k)
            ax.plot(u, y, lw=2, label=r'$\kappa=' + f'{k}$')
            
        ax.axvline(1.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='$u=1$ (transition point)')
        ax.set_title(r"Smooth Cutoff Function $\phi_{\kappa}(u)$", fontsize=18)
        ax.set_xlabel(r"$u$", fontsize=14)
        ax.set_ylabel(r"$\phi_{\kappa}(u)$", fontsize=14)
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        
        filename = f"plot_cutoff_function.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated cutoff function plot to '{filename}'.")
        plt.close(fig)
        
    def generate_P_tau_plot(self):
        """Visualizes the smooth divisor-counting analogue 𝒫_τ(x)."""
        x_min, x_max = self.config.PLOT_P_TAU_RANGE
        k = self.config.STEEPNESS_K
        x = np.linspace(x_min, x_max, 8000)
        y = np.vectorize(P_tau_fast)(x, k)

        fig, ax = plt.subplots(figsize=self.config.PLOT_TAU_SIGMA_FIGSIZE)
        ax.plot(x, y, lw=1.5, color='teal')
        ax.axhline(0, color='black', lw=0.7)
        primes_in_range = sorted([p for p in self.primes if x_min <= p <= x_max])
        for p in primes_in_range:
            ax.axvline(p, color='green', linestyle=':', lw=1.5, alpha=0.7)
            
        ax.set_title(r"Smooth Divisor-Counting Analogue $\mathcal{P}_{\tau}(x)$", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}_{\tau}(x)$", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(primes_in_range)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12, rotation=90)
        
        filename = f"plot_P_tau.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated P_tau plot to '{filename}'.")
        plt.close(fig)
        
    def generate_P_sigma_plot(self):
        """Visualizes 𝒫_σ(x) in a range where non-integer zeros are queried."""
        x_min, x_max = self.config.PLOT_P_SIGMA_RANGE
        k = self.config.STEEPNESS_K
        x = np.linspace(x_min, x_max, 10000)
        y = np.vectorize(P_sigma_fast)(x, k)

        fig, ax = plt.subplots(figsize=self.config.PLOT_SIGMA_ZERO_FIGSIZE)
        ax.plot(x, y, lw=1.5, color='indigo')
        ax.axhline(0, color='black', lw=0.7, label='y = 0')
        
        primes_in_range = sorted([p for p in self.primes if x_min <= p <= x_max])
        for p in primes_in_range:
            ax.axvline(p, color='green', linestyle=':', lw=1.5, alpha=0.7, label=f'$x={p}$ (prime)')
            
        ax.set_title(r"Smooth Divisor-Sum Analogue $\mathcal{P}_{\sigma}(x)$ and its Zeros", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}_{\sigma}(x)$", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1, 1) # Focus on the vertical axis to highlight zero-crossings.
        ax.tick_params(axis='both', labelsize=12)
        
        # Consolidate legend for clarity
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        
        filename = f"plot_P_sigma_zeros.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated P_sigma zero-behavior plot to '{filename}'.")
        plt.close(fig)

    def generate_companion_zeros_plot(self):
        """
        Companion zeros illustration: show P_tau(x; kappa) for several kappa values
        over a small x-range, with a few small primes highlighted.
        """
        x_min, x_max = 2.0, 8.0
        k_values = [2.0, 5.0, 10.0, 100.0]
        primes_mark = [3, 5, 7]

        x = np.linspace(x_min, x_max, 4000)
        fig, ax = plt.subplots(figsize=(12, 6))

        for kappa in k_values:
            y = np.vectorize(P_tau_fast)(x, kappa)
            ax.plot(x, y, lw=1.0, label=rf"$\mathcal{{P}}_\tau(x;\,\kappa={int(kappa)})$")

        # Mark a few small primes
        for j, p in enumerate(primes_mark):
            ax.axvline(p, linestyle="--", alpha=0.5, lw=1.0,
                       label=r"odd primes" if j == 0 else None)

        ax.set_title(r"Companion zeros (small $\kappa$): $\kappa=2,5,10,100$, primes $3,5,7$", fontsize=16)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}_{\tau}(x;\kappa)$", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.legend()
        filename = f"plot_companion_zeros.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated companion zeros plot to '{filename}'.")
        plt.close(fig)

    def generate_companion_zeros_sigma_plot(self):
        """
        Companion zeros illustration for P_sigma(x; kappa) on a small range,
        showing the asymmetric pair near odd primes (left zero exponentially close,
        right zero visible), cf. Conjecture~Psigma-companions.
        """
        x_min, x_max = 2.0, 8.0
        k_values = [2.0, 10.0, 20.0, 100.0]
        primes_mark = [3, 5, 7]

        x = np.linspace(x_min, x_max, 4000)
        fig, ax = plt.subplots(figsize=(12, 6))

        for kappa in k_values:
            y = np.vectorize(P_sigma_fast)(x, kappa)
            ax.plot(x, y, lw=1.0, label=rf"$\mathcal{{P}}_\sigma(x;\,\kappa={int(kappa)})$")

        # Mark a few small primes
        for j, p in enumerate(primes_mark):
            ax.axvline(p, linestyle="--", alpha=0.5, lw=1.0,
                       label=r"odd primes" if j == 0 else None)

        ax.set_title(r"Companion zeros (small $\kappa$) for $\mathcal{P}_\sigma$: $\kappa=2,10,20,100$, primes $3,5,7$", fontsize=16)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}_{\sigma}(x;\kappa)$", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.legend()
        filename = f"plot_companion_zeros_sigma.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"   Generated sigma companion zeros plot to '{filename}'.")
        plt.close(fig)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Instantiate the main classes with the shared configuration
    config = Config()
    verifier = Verifier(config)
    plotter = Plotter(config, verifier.primes)

    # set working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Output directory set to script's location: ", script_dir)
    
    # Additional benchmark: non-accumulative H-variant vs constant-C baseline
    # uncomment to run  
    """  
    run_prime_counting_H_benchmark(verifier.primes, config, x_max=10000,
                               alpha_grid=(18.4, 18.45, 18.5, 18.55, 18.6, 18.65, 18.7, 18.75, 18.8),
                               gamma_grid=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
                               c_grid=(0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
                               C_baseline=1e-3)
    """
    # Generate all plots for the paper
    plotter.run_all_plots()

    # Run all verification tests for the foundational function
    verifier.run_all_tests()

    print("\nScript finished successfully.")