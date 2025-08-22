from pandas import DataFrame
from shapely.geometry import Polygon

from tools.figures import create_estimate_figure
from tools.generate import (
    INSCRIBED_CIRCLE,
    SQUARE,
    generate_estimates_polars,
)


def monte_carlo_example_polars(
    square_polygon: Polygon,
    inscribed_circle_polygon: Polygon,
    n_samples: int,
    seed: int,
    file: str | None,
    x_label: str = "observations",
    y_label: str = "pi_estimate",
) -> None:
    """
    Run a Monte Carlo simulation to estimate `pi` using a Polars LazyFrame pipeline for performance
    and optionally save the results as an interactive HTML plot.

    This function wraps the Polars LazyFrame estimation logic, generates a DataFrame
    of running `pi` estimates, and saves a scatter plot of those estimates if a file
    path is provided.

    Parameters:
        square_polygon (Polygon): A Shapely Polygon whose bounding box is used for sampling.
        inscribed_circle_polygon (Polygon): A Shapely Polygon representing the circle inscribed within the square.
        n_samples (int): Number of random points to generate for the simulation.
        seed (int | None, optional): Optional seed for reproducibility. Defaults to None.
        file (str | None): Optional path to save the resulting plot as an HTML file, if None, no plot is saved.
            Defaults to None.
        x_label (str, optional): Label for the x-axis (e.g., number of observations). Defaults to "observations".
        y_label (str, optional): Label for the y-axis (e.g., running estimate of `pi`). Defaults to "pi_estimate".

    Example:
        ```python
        SQUARE = Polygon(((0, 0), (0, 4), (4, 4), (4, 0)))
        INSCRIBED_CIRCLE = SQUARE.centroid.buffer(
            distance=2,  # Radius
            quad_segs=1000,  # Such circular, much wow
        )

        monte_carlo_example_polars(
            square_polygon=SQUARE,
            inscribed_circle_polygon=INSCRIBED_CIRCLE,
            n_samples=1000,
            seed=42,
            file="C:\\Users\\Archimedes\\Downloads\\monte_carlo_polars.html",
            x_label="observations",
            y_label="pi_estimate",
        )
        ```
    """

    estimate_df = generate_estimates_polars(
        square_polygon=square_polygon,
        inscribed_circle_polygon=inscribed_circle_polygon,
        n_samples=n_samples,
        seed=seed,
        x_label=x_label,
        y_label=y_label,
    )

    # Create a plotly figure if file path is given
    if isinstance(file, str):
        assert isinstance(estimate_df, DataFrame)  # Satisfy pyright type checker
        create_estimate_figure(
            file=file,
            df=estimate_df,
            x_label=x_label,
            y_label=y_label,
        )


if __name__ == "__main__":

    print(f"Square Area: {SQUARE.area}")
    print(f"Inscribed Circle Area: {INSCRIBED_CIRCLE.area}")
    pi_ratio = 4 * (INSCRIBED_CIRCLE.area / SQUARE.area)
    print(f"Area Ratio (Circle Area/Square Area): {pi_ratio}")

    monte_carlo_example_polars(
        square_polygon=SQUARE,
        inscribed_circle_polygon=INSCRIBED_CIRCLE,
        n_samples=10000,
        seed=42,
        file=r"src\monte_carlo_example\figs\monte_carlo_example_polars.html",
        x_label="observations",
        y_label="pi_estimate",
    )
