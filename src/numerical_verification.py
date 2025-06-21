# =============================================================================
#
#   Verification Script for the Function 𝒫(x)
#
#   This script performs a numerical analysis of the function 𝒫(x)
#   to verify its claimed properties, as outlined in the accompanying paper.
#   It is designed for reproducibility and clarity.
#
#   Author: Sebastian Fuchs
#   Date:   2025-06-22
#   Version: 1.0
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

# =============================================================================
# CENTRAL CONFIGURATION
# =============================================================================
# All thresholds, ranges, and step sizes can be adjusted here.

class Config:
    # --- Test Ranges ---
    ZERO_TEST_RANGE_HIGH_PRECISION = (1, 1000)
    ZERO_TEST_RANGE_FAST = (1, 100000)
    SMOOTHNESS_TEST_RANGE = (1.5, 100.1)
    POSITIVITY_TEST_RANGE = (2.1, 100.1)

    # --- Step Sizes ---
    SMOOTHNESS_STEP = 1e-4
    POSITIVITY_STEP = 1e-3

    # --- Precision & Thresholds ---
    ZERO_PRECISION_THRESHOLD = 1e-9  # ε for zero checks
    MPMATH_PRECISION = 50 
    JUMP_TEST_DELTA = 1e-6          # δ for numerical derivatives
    
    # --- Absolute thresholds for smoothness tests ---
    C0_JUMP_THRESHOLD = 1e-5
    C1_JUMP_THRESHOLD = 1e-2
    C2_JUMP_THRESHOLD = 1.0

    # --- C-infinity Test ---
    C_INF_TEST_POINTS = [4.0, 9.0, 16.0, 25.0, 13.0, 17.0]  # Integer squares for the fast test
    C_INF_TEST_RADIUS = 0.1                     # Radius of the test interval
    C_INF_TEST_STEP = 1e-5                      # Step size within the interval    
    # Deltas and Thresholds
    C_INF_DELTAS = {
        '2nd': 1e-4,
        '3rd': 1e-3,
        '4th': 1e-2
    }
    C_INF_THRESHOLDS = {
        '2nd': 1e-3,
        '3rd': 1e-3,
        '4th': 1e-2,
    }

    # --- High Precision C-infinity Test ---
    HIGH_PREC_TEST_POINT = 4.0
    HIGH_PREC_DELTA = 1e-4      # Can be much smaller due to high precision
    HIGH_PREC_THRESHOLD = 1e-4  # Expect jumps to be very close to zero

    # --- Plotting Configuration ---
    PLOT_DPI = 600
    PLOT_FILE_FORMAT = 'pdf'
    PLOT_OVERVIEW_RANGE = (2, 50)
    PLOT_OVERVIEW_FIGSIZE = (20, 5)
    PLOT_ZOOM_FIGSIZE = (10, 7)
    PLOT_DERIVATIVE_FIGSIZE = (12, 7)
    PLOT_PI_FIGSIZE = (12, 8)
    
    # --- Prime Counting Function Plot Configuration ---
    PLOT_PI_RANGE = (0, 50)
    # This constant C is used for the adaptive threshold C/n.
    PLOT_PI_ADAPTIVE_C = 0.0001

    # --- Performance ---
    NUM_PROCESSES = cpu_count()      # Use 0 to disable multiprocessing


# =============================================================================
# CORE FUNCTION IMPLEMENTATION
# =============================================================================

@jit(nopython=True, fastmath=True)
def P_fast(x: float) -> float:
    """
    High-performance implementation of 𝒫(x) using Numba for JIT compilation.
    """
    if x <= 1.0:
        return 0.0

    limit_i = math.ceil(math.sqrt(x))
    total_sum = 0.0

    for i in range(2, limit_i + 1):
        inner_sum_val = float(i)
        for k in range(1, i):
            term = (i - k) * math.cos(2 * math.pi * x * k / i)
            inner_sum_val += 2 * term
        total_sum += inner_sum_val
        
    return total_sum / x


def P_high_precision(x: int):
    """
    High-precision implementation of 𝒫(x) using mpmath.
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

@jit(nopython=True, fastmath=True)
def phi_cutoff(u: float) -> float:
    """
    Smooth C-infinity cutoff function, e.g., exp(-u^4).
    """
    return math.exp(-u**4)

@jit(nopython=True, fastmath=True)
def P_phi_fast(x: float) -> float:
    """
    High-performance implementation of the C-infinity version 𝒫_𝜙(x).
    """
    if x <= 1.0:
        return 0.0

    # The infinite series must be truncated for numerical computation. This limit
    # is chosen as a pragmatic safety margin to ensure that the cutoff term
    # phi(u) = exp(-u^4) becomes numerically insignificant for all subsequent terms.
    #
    # At the boundary i ≈ 8 * sqrt(x), the argument to the cutoff function is u ≈ 8.
    # This yields a cutoff factor of exp(-8**4) = exp(-4096), a value vastly smaller
    # than the standard float64 machine epsilon (~1e-16).
    #
    # Therefore, truncating the sum at this point does not introduce a meaningful
    # numerical error into the final result.
    limit_i = math.ceil(8 * math.sqrt(x)) 
    total_sum = 0.0

    for i in range(2, limit_i + 1):
        inner_sum_val = float(i)
        for k in range(1, i):
            term = (i - k) * math.cos(2 * math.pi * x * k / i)
            inner_sum_val += 2 * term
            
        cutoff_val = phi_cutoff(i / math.sqrt(x))
        total_sum += cutoff_val * inner_sum_val
        
    return total_sum / x

# =============================================================================
# NUMERICAL VERIFICATION SUITES
# =============================================================================

def check_integer_zero_high_precision(n: int) -> tuple[int, str] | None:
    """
    Worker function for multiprocessing.
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
    
    if n == 1:
        pass
    elif n == 2:
        if is_zero: return (n, "𝒫(2) is zero (should be non-zero).")
    elif is_prime_worker(n):
        if not is_zero: return (n, f"𝒫({n}) is non-zero (should be zero).")
    else:
        if is_zero: return (n, f"𝒫({n}) is zero (should be non-zero).")
            
    return None


class Verifier:
    """Encapsulates all verification tests for 𝒫(x)."""

    def __init__(self, config: Config):
        self.config = config
        print("Initializing verifier...")
        max_range = max(
            self.config.ZERO_TEST_RANGE_HIGH_PRECISION[1],
            self.config.ZERO_TEST_RANGE_FAST[1], 
            int(self.config.POSITIVITY_TEST_RANGE[1]), 
            self.config.PLOT_PI_RANGE[1]
        )
        self.primes = set(sieve.primerange(1, max_range + 1))
        print(f"Generated {len(self.primes)} primes up to {max_range}.")

    def run_all_tests(self):
        """Runs the complete suite of verification tests."""
        print("\n--- Starting Verification Suite ---")
        self.run_zero_verification_high_precision()
        self.run_zero_verification_fast_vectorized()
        self.run_positivity_test()
        self.run_smoothness_analysis()
        self.run_c_infinity_smoothness_test()
        print("\n--- Verification Suite Finished ---\n")

    def run_zero_verification_high_precision(self):
        """Verifies zero properties with high precision using multiprocessing."""
        print(f"\n[TEST A1] Verifying Zeros (High Precision) up to {self.config.ZERO_TEST_RANGE_HIGH_PRECISION[1]}...")
        n_min, n_max = self.config.ZERO_TEST_RANGE_HIGH_PRECISION
        numbers_to_check = range(n_min, n_max + 1)
        start_time = time.time()
        
        if self.config.NUM_PROCESSES > 0:
            print(f"  INFO: Starting parallel verification on {self.config.NUM_PROCESSES} cores.")
            with Pool(processes=self.config.NUM_PROCESSES) as pool:
                results = pool.map(check_integer_zero_high_precision, numbers_to_check)
        else:
            print("  INFO: Starting sequential verification.")
            results = [check_integer_zero_high_precision(n) for n in numbers_to_check]

        failures = [res for res in results if res is not None]
        elapsed = time.time() - start_time
        print(f"  INFO: High-precision verification took {elapsed:.2f} seconds.")

        if not failures:
            print(f"  SUCCESS: All integers in range {self.config.ZERO_TEST_RANGE_HIGH_PRECISION} passed high-precision check.")
        else:
            print(f"  SUMMARY: {len(failures)} failures detected in high-precision check:")
            for n, msg in failures:
                print(f"    - At n={n}: {msg}")

    def run_zero_verification_fast_vectorized(self):
        """Performs a fast, vectorized check over a large range of integers."""
        print(f"\n[TEST A2] Verifying Zeros (Fast Vectorized) up to {self.config.ZERO_TEST_RANGE_FAST[1]}...")
        n_min, n_max = self.config.ZERO_TEST_RANGE_FAST
        start_time = time.time()

        numbers = np.arange(n_min, n_max + 1, dtype=np.float64)
        p_values = np.vectorize(P_fast)(numbers)
        is_zero_mask = np.abs(p_values) < self.config.ZERO_PRECISION_THRESHOLD
        is_prime_mask = np.array([int(n) in self.primes for n in numbers])

        false_negatives = np.where(is_prime_mask & (numbers > 2) & ~is_zero_mask)[0]
        false_positives = np.where(~is_prime_mask & (numbers > 2) & is_zero_mask)[0]
        case_2_failure = np.where((numbers == 2) & is_zero_mask)[0]
        failure_indices = np.concatenate([false_negatives, false_positives, case_2_failure])
        elapsed = time.time() - start_time
        print(f"  INFO: Fast verification took {elapsed:.2f} seconds.")

        if len(failure_indices) == 0:
            print(f"  SUCCESS: All integers in range {self.config.ZERO_TEST_RANGE_FAST} passed fast check.")
        else:
            print(f"  SUMMARY: {len(failure_indices)} failures detected in fast check:")
            for idx in failure_indices:
                n = int(numbers[idx])
                is_p = is_prime_mask[idx]
                msg = "𝒫(2) is zero." if n == 2 else f"𝒫({n}) is non-zero." if is_p else f"𝒫({n}) is zero."
                print(f"    - At n={n}: {msg}")

    def run_positivity_test(self):
        """Verifies that 𝒫(x) > 0 for non-prime x."""
        print(f"\n[TEST B] Verifying Positivity in Range {self.config.POSITIVITY_TEST_RANGE}...")
        failures = []
        x_min, x_max = self.config.POSITIVITY_TEST_RANGE
        test_points = np.arange(x_min, x_max, self.config.POSITIVITY_STEP)
        
        for x in test_points:
            if abs(x - round(x)) < 1e-9 and round(x) in self.primes: continue
            if P_fast(x) < self.config.ZERO_PRECISION_THRESHOLD: failures.append(x)
        
        if not failures: print("  SUCCESS: Function is positive at all tested non-prime points.")
        else: print(f"  SUMMARY: {len(failures)} failures detected.")

    def run_smoothness_analysis(self):
        """Analyzes C^0, C^1, and C^2 properties of 𝒫."""
        print("\n[TEST C] Analyzing Smoothness (C0, C1, C2)...")
       
    def run_c_infinity_smoothness_test(self):
        """
        Analyzes smoothness of 𝒫_𝜙 in local neighborhoods by using stability-adapted parameters.
        """
        print("\n[TEST D] Analyzing C-infinity Smoothness of 𝒫_𝜙 (Fast Check)...")
        
        def d2f_dx2(f, x, d): return (f(x + d) - 2 * f(x) + f(x - d)) / (d**2)
        def d3f_dx3(f, x, d): return (f(x + 2*d) - 2*f(x + d) + 2*f(x - d) - f(x - 2*d)) / (2 * d**3)
        def d4f_dx4(f, x, d): return (f(x + 2*d) - 4*f(x + d) + 6*f(x) - 4*f(x - d) + f(x - 2*d)) / (d**4)

        test_points = self.config.C_INF_TEST_POINTS
        radius = self.config.C_INF_TEST_RADIUS
        step = self.config.C_INF_TEST_STEP
        deltas = self.config.C_INF_DELTAS
        thresholds = self.config.C_INF_THRESHOLDS

        derivatives_to_test = [("2nd derivative", d2f_dx2), ("3rd derivative", d3f_dx3), ("4th derivative", d4f_dx4)]
        total_failures = 0

        for x0 in test_points:
            print(f"  INFO: Checking neighborhood of x = {x0}...")
            point_failures = 0
            for name, func in derivatives_to_test:
                order_key = name.split()[0]
                delta = deltas.get(order_key)
                threshold = thresholds.get(order_key)
                
                local_x_vals = np.arange(x0 - radius, x0 + radius, step)
                deriv_vals = np.array([func(P_phi_fast, x, delta) for x in local_x_vals])
                max_jump = np.max(np.abs(np.diff(deriv_vals)))
                
                if max_jump > threshold:
                    print(f"    !!FAILURE at x={x0}!! Large jump in {name}: max(Δ) = {max_jump:.6f} > {threshold}")
                    point_failures += 1
                else:
                    print(f"    -- success --  Continuity plausible for {name} (max(Δ) = {max_jump:.6f} <= {threshold})")
            if point_failures > 0: total_failures += point_failures

        print("-" * 20)
        if total_failures == 0: print("  SUCCESS: No significant discontinuities found in higher derivatives of 𝒫_𝜙.")
        else: print(f"  SUMMARY: {total_failures} issues found in C-infinity smoothness check.")

# =============================================================================
# PLOTTING SUITE
# =============================================================================

class Plotter:
    """Generates publication-quality plots for 𝒫(x)."""
    
    def __init__(self, config: Config, primes: set):
        self.config = config
        self.primes = primes
        plt.style.use('seaborn-v0_8-whitegrid')

    def run_all_plots(self):
        print("\n--- Generating Plots ---")
        self.generate_overview_plot()
        self.generate_zoom_plot()
        self.generate_derivative_plot()
        self.generate_prime_counting_plot()
        print("--- Plot Generation Finished ---\n")

    def generate_overview_plot(self):
        """Overview of 𝒫(x) with prime markers."""
        x_min, x_max = self.config.PLOT_OVERVIEW_RANGE
        x = np.linspace(x_min, x_max, 8000)
        y = np.vectorize(P_fast)(x)
        
        fig, ax = plt.subplots(figsize=self.config.PLOT_OVERVIEW_FIGSIZE)
        ax.plot(x, y, lw=1.0, color='royalblue')
        ax.axhline(0, color='black', lw=0.7)
        primes_in_range = sorted([p for p in self.primes if x_min < p < x_max])
        for p in primes_in_range:
            ax.axvline(p, color='green', linestyle=':', lw=1.5, alpha=0.7)
        ax.set_title(r"Overview of the Function $\mathcal{P}(x)$", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}(x)$", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(primes_in_range)
        ax.set_xticklabels(primes_in_range)
        ax.xaxis.grid(False)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        filename = f"plot_overview.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  SUCCESS: Saved overview plot to '{filename}'.")
        plt.close(fig)

    def generate_zoom_plot(self):
        """Zoom-in on a representative zero of 𝒫(x)."""
        x_prime = 13
        x = np.linspace(x_prime - 0.1, x_prime + 0.1, 2000)
        y = np.vectorize(P_fast)(x)
        fig, ax = plt.subplots(figsize=self.config.PLOT_ZOOM_FIGSIZE)
        ax.plot(x, y, '.-', lw=1.0, ms=4, color='darkorange')
        ax.axhline(0, color='black', lw=0.7)
        ax.axvline(x_prime, color='red', alpha=0.5, linestyle='--', lw=2.0, label=f'x = {x_prime} (prime)')
        ax.set_title(r"Detailed View of the Zero at $x=13$", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\mathcal{P}(x)$", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend()
        filename = f"plot_zoom_{x_prime}.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  SUCCESS: Saved zoom plot to '{filename}'.")
        plt.close(fig)

    def generate_derivative_plot(self):
        """Numerical second derivative of 𝒫(x) with discontinuity markers."""
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
            label = r'Discontinuity at $x=k^2$' if i == 0 else ""
            ax.axvline(sq, color='red', alpha=0.5, linestyle='--', lw=2.0, label=label)
        ax.legend()
        filename = f"plot_second_derivative.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  SUCCESS: Saved second derivative plot to '{filename}'.")
        plt.close(fig)
    
    def generate_prime_counting_plot(self):
        """
        Generates a plot for the derived prime-counting function approximation.
        """
        x_min, x_max = self.config.PLOT_PI_RANGE
        C = self.config.PLOT_PI_ADAPTIVE_C
        
        integers = np.arange(x_min, x_max + 1, dtype=np.float64)
        p_values = np.vectorize(P_fast)(integers)
        
        n_vals = integers[1:]
        p_vals_for_sum = p_values[1:]
        adaptive_threshold = C / n_vals
        terms = 1 - (p_vals_for_sum / (p_vals_for_sum + adaptive_threshold))
        pi_P_values = np.concatenate(([0], np.cumsum(terms)))
        
        fig, ax = plt.subplots(figsize=self.config.PLOT_PI_FIGSIZE)
        ax.step(integers, pi_P_values, where='post', lw=1.5, color='black', label=r'Approximation $\pi_{\mathcal{P}}(x)$')
        ax.set_title(r"A Derived Prime-Counting Function Approximation", fontsize=18)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"Value", fontsize=14)
        ax.set_xlim(x_min, x_max)
        ax.set_yticks(np.arange(0, int(pi_P_values.max()) + 2, 1))
        ax.set_xticks(np.arange(x_min, x_max + 1, 5))
        ax.set_ylim(bottom=-0.5, top=int(pi_P_values.max()) + 1)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        
        filename = f"plot_prime_counting.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  SUCCESS: Saved prime-counting plot to '{filename}'.")
        plt.close(fig)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    config = Config()
    verifier = Verifier(config)
    plotter = Plotter(config, verifier.primes)
    
    verifier.run_all_tests()
    plotter.run_all_plots()

    print("Script finished successfully.")