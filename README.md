# From Trial Division to Smooth Prime Indicators: A Framework Based on Fourier Series

**Author:** Sebastian Fuchs  
**Email:** [sebastian.fuchs@hu-berlin.de](mailto:sebastian.fuchs@hu-berlin.de)  
**Location:** Berlin, Germany  
**Affiliation:** Humboldt University of Berlin  
**GitHub:** [SebastianFoxxx/analytic-prime-indicator](https://github.com/SebastianFoxxx/analytic-prime-indicator)  
**Paper on arXiv:** [https://arxiv.org/abs/2506.18933]()  
**ORCID:** [0009-0009-1237-4804](https://orcid.org/0009-0009-1237-4804)  
**DOI:** [10.5281/zenodo.15712807](https://doi.org/10.5281/zenodo.15712807)  
**Date:** 2025-06-26  
**Version:** 1.1.0  

---

### Abstract

This work introduces a flexible framework for constructing smooth ($C^\infty$) analogues of classical arithmetic functions, based on the smoothing of a trigonometric representation of trial division. A foundational function $\mathcal{P}\colon\mathbb{R}\to\mathbb{R}$ of class $C^1$ is first presented, whose zeros for $x>2$ correspond precisely to the odd primes. The fact that this function is of class $C^1$ but not smoother motivates its generalization into a versatile framework for constructing fully smooth ($C^\infty$) analogues. This framework is then applied to construct two novel functions: a smooth analogue of the divisor-counting function, $\mathcal{P}_{\tau}$, and a smooth analogue of the sum-of-divisors function, $\mathcal{P}_{\sigma}$. Both functions are proven to be of class $C^\infty$. It is shown that these new constructions possess a complete prime-zero property for all integers $x \ge 2$. The robustness of this property for real numbers is analyzed for each function. It is demonstrated that $\mathcal{P}_{\tau}$ provides a robust prime indicator for all real numbers, while the properties of $\mathcal{P}_{\sigma}$ in this regard lead to an open question concerning the existence of non-integer zeros. The construction methodology, properties, and potential of these functions are discussed in detail.

### Repository Contents

This repository provides all the necessary materials to reproduce the results and figures presented in the paper. The structure is as follows:

* **/paper/**: Contains the LaTeX source code (`analytic_prime_indicator.tex`) and all figures for the main manuscript.
* **/code/**: Contains the primary Python script for verification and plotting.
    * `numerical_verification.py`: The script to run all numerical tests for the functions $\mathcal{P}(x)$, $\mathcal{P}_{\tau}(x)$, and $\mathcal{P}_{\sigma}(x)$, and to generate all figures presented in the paper.
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
    git clone [https://github.com/SebastianFoxxx/analytic-prime-indicator.git](https://github.com/SebastianFoxxx/analytic-prime-indicator.git)
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

4.  **Run the Verification Script** Execute the main Python script. This will perform all numerical checks described in the script and regenerate all figures (`.pdf` files) in the root directory of the .py file.
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