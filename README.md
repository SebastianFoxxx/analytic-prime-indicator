# An Analytic Prime Indicator Based on the Fejér Kernel

**Author:** Sebastian Fuchs  
**Email:** [sebastian.fuchs@hu-berlin.de](sebastian.fuchs@hu-berlin.de)  
**Location:** Berlin, Germany  
**Affiliation:** Humboldt University of Berlin  
**GitHub:** [SebastianFoxxx](https://github.com/SebastianFoxxx/analytic-prime-indicator)  
**Paper on arXiv:** [Link to be added upon submission to arXiv.org]()  
**ORCID:** [0009-0009-1237-4804](https://orcid.org/0009-0009-1237-4804)  
**DOI:** [10.5281/zenodo.15712807](https://doi.org/10.5281/zenodo.15712807)  
**Date:** 2025-06-22  
**Version:** 1.0.0  

---

### Abstract

This note introduces an analytic prime indicator, constructed by smoothing a trigonometric analogue of trial division. First, a function $\mathcal{P}\colon\mathbb{R}\to\mathbb{R}$ of class $C^1$ is presented, whose zeros for $x>2$ correspond precisely to the odd primes. Its second derivative exhibits jump discontinuities at integer squares, arising from the ceiling function in the summation bound. Subsequently, it is shown how this construction can be modified via an infinite series to yield a globally smooth function, $\mathcal{P}_{\phi}$, of class $C^\infty$ that preserves this prime-zero property.

### Repository Contents

This repository provides all the necessary materials to reproduce the results and figures presented in the paper. The structure is as follows:

* **/paper/**: Contains the LaTeX source code (`analytic_prime_indicator.tex`) and all figures for the main manuscript.
* **/code/**: Contains the primary Python script for verification and plotting.
    * `numerical_verification.py`: The script to run all numerical tests and generate the figures.
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
    cd YourRepoName
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

4.  **Run the Verification Script** Execute the main Python script. This will perform all numerical checks described in the script and regenerate all figures (`.pdf` files) in the root directory of the project.
    ```bash
    python code/numerical_verification.py
    ```
    The script will print its progress to the console.

5.  **Compile the LaTeX Documents** To generate the PDFs from the source, compile the `.tex` files using your LaTeX distribution. You may need to run the command twice for cross-references to be resolved correctly.
    ```bash
    # To compile the main paper
    pdflatex -output-directory=paper paper/analytic_prime_indicator.tex    
    ```

### License

The source code in this repository is released under the **MIT License**. See the `LICENSE` file for more details. The content of the research papers is subject to the copyright of the author.