# Fejér--Kernel Prime Indicators

**Author:** Sebastian Fuchs  
**Email:** [sebastian.fuchs@hu-berlin.de](mailto:sebastian.fuchs@hu-berlin.de)  
**Location:** Berlin, Germany  
**Affiliation:** Humboldt University of Berlin  
**GitHub:** [SebastianFoxxx/analytic-prime-indicator](https://github.com/SebastianFoxxx/analytic-prime-indicator)  
**Paper on arXiv:** [https://arxiv.org/abs/2506.18933](https://arxiv.org/abs/2506.18933)  
**ORCID:** [0009-0009-1237-4804](https://orcid.org/0009-0009-1237-4804)  
**DOI:** [10.5281/zenodo.15748475](https://doi.org/10.5281/zenodo.15748475)  
**Date:** 2025-10-15  
**Version:** 1.2.0

-----

### Abstract

A $C^1$ prime indicator $\mathcal{P}: \mathbb{R} \to \mathbb{R}$ is constructed by applying the Fejér identity to the sine–quotient encoder of trial division. For integers $n\ge 2$, $\mathcal P(n)=0$ holds exactly for odd primes; $\mathcal P(2)>0$. For all non-integers $x>1$ one has $\mathcal P(x)>0$. The function is piecewise $C^\infty$ and its second derivative has jumps precisely at the squares $m^2$, with explicit sizes. Replacing the sharp cut-off by a smooth transition yields $C^\infty$ analogues $\mathcal{P}_\tau$ and $\mathcal{P}_\sigma$ with integer limits $\mathcal{P}_\tau(n;\kappa)\to \tau(n)-2$ and $\mathcal{P}_\sigma(n;\kappa)\to \sigma(n)-n-1$ as $\kappa\to\infty$, obtained from locally uniform convergence of derivative series. For large $\kappa$, numerical evidence indicates companion zeros near odd primes for $\mathcal{P}_\tau$ and an asymmetric pair for $\mathcal{P}_\sigma$. No assertion is made beyond integer input, and no statements are claimed about the prime number theorem or zero distributions of $L$-functions. The appendix includes two illustrative prime-counting sums.

### Numerical Verification and Plotting Script

The script `numerical_verification.py` is a comprehensive suite for the numerical work presented in the paper. It is designed to be highly configurable, performant, and robust. Its key features include:

  * **Core Implementations:** Provides performant implementations of the foundational function $\mathcal{P}(x)$ and its smooth analogues $\mathcal{P}_{\tau}(x)$ and $\mathcal{P}_{\sigma}(x)$, accelerated with Numba for just-in-time compilation.
  * **Numerical Verification Suite:** A series of rigorous tests to validate the properties of the functions:
      * **High-Precision Zero Checks:** Verifies that $\mathcal{P}(n)=0$ exclusively for odd primes at integer arguments using the arbitrary-precision `mpmath` library.
      * **Positivity Scans:** Numerically confirms that $\mathcal{P}(x) > 0$ for non-integer $x > 1$ on a fine-grained grid.
      * **Consistency Tests:** Ensures that different evaluation methods for the core Fejér term (closed-form, cosine polynomial, and resonant partial fractions) are numerically consistent within specified tolerances.
      * **Companion Zero Validation:** Tests the conjectures regarding the behavior of zeros for $\mathcal{P}_{\tau}(x)$ and $\mathcal{P}_{\sigma}(x)$ near odd primes for varying steepness parameters.
  * **Performance Optimization:** Leverages multiprocessing via Python's `multiprocessing` pool to significantly speed up computationally intensive verification tasks.
  * **Plot Generation:** Automatically generates all figures presented in the paper in publication-quality format (`.pdf` by default). This includes overviews, local zooms, derivative plots, and visualizations of the smooth analogues and their properties.
  * **Central Configuration:** All numerical parameters, test ranges, tolerances, and plot settings can be easily adjusted within the central `Config` class at the top of the script.

### Repository Contents

This repository provides all the necessary materials to reproduce the results and figures presented in the paper. The structure is as follows:

  * **/paper/**: Contains the LaTeX source code (`analytic_prime_indicator.tex`) and all figures for the main manuscript.
  * **/code/**: Contains the primary Python script for verification and plotting.
      * `numerical_verification.py`: The script to run all numerical tests and generate all figures.
      * `requirements.txt`: A list of all required Python packages.
  * `LICENSE`: The license under which the code in this repository is shared (MIT License).
  * `.gitignore`: Specifies files to be ignored by version control.

### How to Reproduce the Results

To run the numerical verifications and regenerate all plots from the paper, please follow these steps.

#### Prerequisites

  * Python 3.8 or newer.
  * A Python package manager, `pip`.
  * A functioning LaTeX distribution (e.g., [TeX Live](https://www.tug.org/texlive/), [MiKTeX](https://miktex.org/)).

#### Step-by-Step Instructions

1.  **Clone the Repository** Clone this repository to your local machine using:

    ```bash
    git clone https://github.com/SebastianFoxxx/analytic-prime-indicator.git
    cd analytic-prime-indicator
    ```

2.  **Set Up a Virtual Environment (Recommended)** It is best practice to create a virtual environment to avoid conflicts with other Python projects.

    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    # python -m venv venv
    # venv\Scripts\activate
    ```

3.  **Install Required Packages** Install all dependencies listed in `requirements.txt` using pip.

    ```bash
    pip install -r code/requirements.txt
    ```

4.  **Run the Verification Script** Execute the main Python script. This will perform all numerical checks described in the script and regenerate all figures (`.pdf` files) in the directory where the script is located.

    ```bash
    python code/numerical_verification.py
    ```

    The script will print its progress to the console.

5.  **Compile the LaTeX Document** To generate the PDF from the source, compile the `.tex` file using your LaTeX distribution. You may need to run the command twice for cross-references to be resolved correctly.

    ```bash
    # To compile the main paper
    pdflatex -output-directory=paper paper/analytic_prime_indicator.tex
    ```

### License

The source code in this repository is released under the **MIT License**. See the `LICENSE` file for more details. The content of the research paper is subject to the copyright of the author.