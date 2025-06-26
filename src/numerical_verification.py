# =============================================================================
#
#   Numerical Verification and Plotting Script for the Framework of
#   Smooth, Analytic Analogues of Divisor Functions
#
#   This script provides a comprehensive suite for numerically verifying the
#   properties of the functions 𝒫(x), 𝒫_τ(x), and 𝒫_σ(x), and for
#   generating all plots presented in the accompanying paper.
#
#   Author: Sebastian Fuchs
#   Date:   2025-06-26
#   Version: 1.1
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
# All thresholds, ranges, and parameters can be adjusted here.

class Config:
    # --- Test Ranges for the foundational function 𝒫(x) ---
    ZERO_TEST_RANGE_HIGH_PRECISION = (1, 1000)
    ZERO_TEST_RANGE_FAST = (1, 100000)
    POSITIVITY_TEST_RANGE = (2.1, 100.1)

    # --- Step Sizes ---
    POSITIVITY_STEP = 1e-3

    # --- Precision & Thresholds ---
    ZERO_PRECISION_THRESHOLD = 1e-9  # ε for zero checks
    MPMATH_PRECISION = 50 
    
    # --- Parameters for the C^∞ Framework ---
    # Steepness parameter 'k' for the phi_mod cutoff function.
    # A larger k makes the cutoff sharper, approximating a step function more closely.
    STEEPNESS_K = 100.0

    # --- Plotting Configuration ---
    PLOT_DPI = 600
    PLOT_FILE_FORMAT = 'pdf'
    
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
    # This constant C is used for the adaptive threshold C/n.
    PLOT_PI_ADAPTIVE_C = 0.0001

    # --- Performance ---
    # Set to 0 to disable multiprocessing for easier debugging.
    NUM_PROCESSES = cpu_count()


# =============================================================================
# CORE FUNCTION IMPLEMENTATIONS
# =============================================================================

# --- Foundational Function 𝒫(x) ---

@jit(nopython=True, fastmath=True)
def P_fast(x: float) -> float:
    """
    High-performance implementation of the foundational function 𝒫(x) using Numba.
    This function has C^1 smoothness.
    """
    if x <= 1.0:
        return 0.0

    limit_i = math.ceil(math.sqrt(x))
    total_sum = 0.0

    for i in range(2, limit_i + 1):
        # This inner loop calculates the Fejér kernel term F(x, i)
        inner_sum_val = float(i)
        for k in range(1, i):
            term = (i - k) * math.cos(2 * math.pi * x * k / i)
            inner_sum_val += 2 * term
        total_sum += inner_sum_val
        
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

# --- C^∞ Framework Components ---

@jit(nopython=True, fastmath=True)
def phi_mod(u: float, k: float) -> float:
    """
    The C^∞ smooth transition cutoff function based on the hyperbolic tangent,
    as defined in the paper. It smoothly transitions from 1 to 0 around u=1.
    """
    return (1.0 - math.tanh(k * (u - 1.0))) / 2.0


@jit(nopython=True, fastmath=True)
def P_tau_fast(x: float, k: float) -> float:
    """
    High-performance implementation of the C^∞ smooth divisor-counting analogue, 𝒫_τ(x).
    """
    if x < 2.0: # The function is defined for x>0, but behavior is trivial/negative for x<2
        # A simple computation for this range can be added if needed,
        # but for plotting purposes, we focus on the prime-indicating region.
        # This implementation will show non-zero values for x<2.
        pass

    # The infinite series must be truncated. The cutoff function phi_mod ensures
    # rapid convergence. The argument to phi_mod is i/(x+1). For i > x+1, the
    # argument is > 1 and the function value drops exponentially.
    # A pragmatic limit like 3*x is more than sufficient for k>=100, as the
    # cutoff term becomes numerically indistinguishable from zero.
    limit_i = math.ceil(3 * x + 20) if x > 1 else 30
    total_sum = 0.0

    for i in range(2, limit_i + 1):
        # Calculate the Fejér kernel term F(x, i)
        fejer_term = float(i)
        for j in range(1, i):
            term = (i - j) * math.cos(2 * math.pi * x * j / i)
            fejer_term += 2 * term

        # Apply the smooth cutoff and the weighting function W(i) = 1/i^2
        cutoff_val = phi_mod(i / (x + 1.0), k)
        total_sum += cutoff_val * (fejer_term / (i * i))
        
    # Final subtraction as per the definition in the paper
    return total_sum - 1.0


@jit(nopython=True, fastmath=True)
def P_sigma_fast(x: float, k: float) -> float:
    """
    High-performance implementation of the C^∞ smooth sum-of-divisors analogue, 𝒫_σ(x).
    """
    if x <= 0:
        return 0.0

    # Similar truncation logic as for P_tau_fast
    limit_i = math.ceil(3 * x + 20) if x > 1 else 30
    total_sum = 0.0

    for i in range(2, limit_i + 1):
        # Calculate the Fejér kernel term F(x, i)
        fejer_term = float(i)
        for j in range(1, i):
            term = (i - j) * math.cos(2 * math.pi * x * j / i)
            fejer_term += 2 * term

        # Apply the smooth cutoff and the weighting function W(i) = 1/i
        cutoff_val = phi_mod(i / (x + 1.0), k)
        total_sum += cutoff_val * (fejer_term / i)
        
    # Final subtraction as per the definition in the paper
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
        print(f"Generated {len(self.primes)} primes up to {max_range}.")

    def run_all_tests(self):
        """Runs the complete suite of verification tests for 𝒫(x)."""
        print("\n--- Starting Verification Suite for 𝒫(x) ---")
        self.run_zero_verification_high_precision()
        self.run_zero_verification_fast_vectorized()
        self.run_positivity_test()
        print("\n--- Verification Suite Finished ---\n")

    def run_zero_verification_high_precision(self):
        """Verifies zero properties of 𝒫(x) with high precision using multiprocessing."""
        print(f"\n[TEST A1] Verifying Zeros of 𝒫(x) (High Precision) up to {self.config.ZERO_TEST_RANGE_HIGH_PRECISION[1]}...")
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
        """Performs a fast, vectorized check of 𝒫(x) over a large range of integers."""
        print(f"\n[TEST A2] Verifying Zeros of 𝒫(x) (Fast Vectorized) up to {self.config.ZERO_TEST_RANGE_FAST[1]}...")
        n_min, n_max = self.config.ZERO_TEST_RANGE_FAST
        start_time = time.time()

        numbers = np.arange(n_min, n_max + 1, dtype=np.float64)
        p_values = np.vectorize(P_fast)(numbers)
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
            print(f"  SUCCESS: All integers > 2 in range {self.config.ZERO_TEST_RANGE_FAST} passed fast check.")
        else:
            print(f"  SUMMARY: {len(failure_indices)} failures detected in fast check:")
            for idx in failure_indices:
                n = int(numbers[idx])
                is_p = is_odd_prime_mask[idx]
                msg = f"𝒫({n}) is non-zero." if is_p else f"𝒫({n}) is zero."
                print(f"    - At n={n}: {msg}")

    def run_positivity_test(self):
        """Verifies that 𝒫(x) > 0 for non-prime x > 2."""
        print(f"\n[TEST B] Verifying Positivity of 𝒫(x) in Range {self.config.POSITIVITY_TEST_RANGE}...")
        failures = []
        x_min, x_max = self.config.POSITIVITY_TEST_RANGE
        test_points = np.arange(x_min, x_max, self.config.POSITIVITY_STEP)
        
        for x in test_points:
            # Skip points that are extremely close to an odd prime, where the value should be zero.
            if abs(x - round(x)) < 1e-9 and round(x) in self.primes and round(x) % 2 != 0: 
                continue
            if P_fast(x) < self.config.ZERO_PRECISION_THRESHOLD: 
                failures.append(x)
        
        if not failures: 
            print("  SUCCESS: Function is positive at all tested non-prime points > 2.")
        else: 
            print(f"  SUMMARY: {len(failures)} failures where P(x) is not positive detected.")
            
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
        # New plots for the framework functions
        self.generate_cutoff_plot()
        self.generate_P_tau_plot()
        self.generate_P_sigma_plot()
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
        print(f"  SUCCESS: Saved foundational function overview plot to '{filename}'.")
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
        print(f"  SUCCESS: Saved zoom plot to '{filename}'.")
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
        print(f"  SUCCESS: Saved second derivative plot to '{filename}'.")
        plt.close(fig)
    
    def generate_prime_counting_plot(self):
        """Generates a plot for the derived prime-counting function approximation."""
        x_min, x_max = self.config.PLOT_PI_RANGE
        C = self.config.PLOT_PI_ADAPTIVE_C
        
        integers = np.arange(x_min, x_max + 1, dtype=np.float64)
        p_values = np.vectorize(P_fast)(integers)
        
        n_vals = integers[1:]
        p_vals_for_sum = p_values[1:]
        adaptive_threshold = C / n_vals
        # The summand is 1 if P(n)=0, and close to 1 otherwise.
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
        
    def generate_cutoff_plot(self):
        """Visualizes the C^∞ smooth cutoff function phi_mod for different k."""
        u_min, u_max = self.config.PLOT_CUTOFF_U_RANGE
        k_values = self.config.PLOT_CUTOFF_K_VALUES
        u = np.linspace(u_min, u_max, 1000)
        
        fig, ax = plt.subplots(figsize=self.config.PLOT_CUTOFF_FIGSIZE)
        for k in k_values:
            y = np.vectorize(phi_mod)(u, k)
            ax.plot(u, y, lw=2, label=f'$k={k}$')
            
        ax.axvline(1.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='$u=1$ (transition point)')
        ax.set_title(r"Smooth Cutoff Function $\phi_{\mathrm{mod}}(u; k)$", fontsize=18)
        ax.set_xlabel(r"$u$", fontsize=14)
        ax.set_ylabel(r"$\phi_{\mathrm{mod}}(u; k)$", fontsize=14)
        ax.set_xlim(u_min, u_max)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        
        filename = f"plot_cutoff_function.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  SUCCESS: Saved cutoff function plot to '{filename}'.")
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
        print(f"  SUCCESS: Saved P_tau plot to '{filename}'.")
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
        ax.set_ylim(-1, 1) # Zoom in vertically to see the zeros clearly
        ax.tick_params(axis='both', labelsize=12)
        
        # Consolidate legend for clarity
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        
        filename = f"plot_P_sigma_zeros.{self.config.PLOT_FILE_FORMAT}"
        plt.savefig(filename, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        print(f"  SUCCESS: Saved P_sigma zero-behavior plot to '{filename}'.")
        plt.close(fig)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Instantiate the main classes with the shared configuration
    config = Config()
    verifier = Verifier(config)
    plotter = Plotter(config, verifier.primes)
    
    # Run all verification tests for the foundational function
    verifier.run_all_tests()
    
    # Generate all plots for the paper
    plotter.run_all_plots()

    print("\nScript finished successfully.")