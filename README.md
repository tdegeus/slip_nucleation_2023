[![DOI](https://zenodo.org/badge/182052974.svg)](https://zenodo.org/doi/10.5281/zenodo.10723197)

[![ci](https://github.com/tdegeus/slip_nucleation_2023/workflows/CI/badge.svg)](https://github.com/tdegeus/slip_nucleation_2023/actions)
[![Documentation Status](https://readthedocs.org/projects/slip-nucleation-2023/badge/?version=latest)](https://slip-nucleation-2023.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://github.com/tdegeus/slip_nucleation_2023/workflows/pre-commit/badge.svg)](https://github.com/tdegeus/slip_nucleation_2023/actions)

**Documentation: [slip-nucleation-2023.readthedocs.io](https://slip-nucleation-2023.readthedocs.io)**

# slip_nucleation_2023

Interface between data and [FrictionQPotFEM](https://github.com/tdegeus/FrictionQPotFEM).

Getting started:

1.  Install dependencies.
    For example using Conda:

    ```bash
    mamba env update --file environment.yaml
    ```

    Tip: use a new environment.
    Note: some of these dependences are needed, some are merely a convenience for running the simulations.

2.  Install this package:

    ```bash
    python -m pip install . -v --no-build-isolation --no-deps
    ```

3.  Build the docs:

    ```bash
    cd docs
    make html
    ```

    Open `docs/_build/html/index.html` in a web browser.

**Warning:** This is a research code. The API and data structure may be subjected to changes.
