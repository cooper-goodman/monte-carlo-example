# monte-carlo-example

A simple example of Monte Carlo estimation of pi using geometric sampling.

## Features

- Estimate pi using random points inside a square and inscribed circle.
- Two implementations:
  - Pure Python for-loop based (`generate_estimates_for_loop`)
  - Vectorized Polars lazy-frame based (`generate_estimates_polars`)
- Easy creation of interactive scatter plots showing pi convergence.
- Simple wrapper functions for quick experiments.

## Installation

### 1. Clone the repository

```bash
cd C:/path/to/install/directory
git clone https://github.com/cooper-goodman/monte-carlo-example.git
cd monte-carlo-example
```

## 2. Install dependencies

### 2.1. pixi

This project was built with [pixi](https://pixi.sh/latest/) a modern python package manager.

If `pixi` is installed it should automatically detect the `pixi.toml` and download the necessary packages.

```bash
cd "C:/path/to/install/directory/monte-carlo-example"
pixi install
```

### 2.2. pip

Dependencies can also be installed with `pip`.

```bash
pip install pandas shapely plotly polars pyarrow matplotlib
```
