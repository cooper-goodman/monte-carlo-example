import polars as pl
from shapely.geometry import Polygon

from tools.figures import create_estimate_figure
from tools.generate import (
    INSCRIBED_CIRCLE,
    SQUARE,
    create_seed_list,
    generate_estimates_polars,
)


def multiple_monte_carlo_example_polars(
    square_polygon: Polygon,
    inscribed_circle_polygon: Polygon,
    n_samples: int,
    n_simulations: int,
    seed: int,
    file: str | None,
    x_label: str = "observations",
    y_label: str = "pi_estimate",
) -> None:
    """
    Run multiple Monte Carlo simulations and average their outputs to estimate `pi` using a Polars LazyFrame
    pipeline for performance, optionally save the results as an interactive HTML plot.

    This function wraps the Polars LazyFrame estimation logic, generates a DataFrame
    of running `pi` estimates, and saves a scatter plot of those estimates if a file
    path is provided.

    Parameters:
        square_polygon (Polygon): A Shapely Polygon whose bounding box is used for sampling.
        inscribed_circle_polygon (Polygon): A Shapely Polygon representing the circle inscribed within the square.
        n_samples (int): Number of random points to generate for the simulation.
        n_simulations (int): Number of separate simulations to generate who's outputs will be averaged.
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
            n_simulations=10,
            seed=42,
            file="C:\\Users\\Archimedes\\Downloads\\monte_carlo_polars.html",
            x_label="observations",
            y_label="pi_estimate",
        )
        ```
    """

    # Create a list of random seeds given the optional main random seed
    seed_list = create_seed_list(size=n_simulations, seed=seed)

    # Initialize a list to hold intermediate LazyFrames
    estimates_df_list = []

    # For each seed run a separate random simulation
    for idx, s in enumerate(seed_list):

        sim_title = f"Simulation #{idx + 1}"
        print(f"{sim_title} (seed={s})")

        estimate_df = generate_estimates_polars(
            square_polygon=square_polygon,
            inscribed_circle_polygon=inscribed_circle_polygon,
            n_samples=n_samples,
            seed=s,
            x_label=x_label,
            y_label=y_label,
            return_lazy_frame=True,  # Return intermediate LazyFrame
        )

        assert isinstance(estimate_df, pl.LazyFrame)

        # Keep only necessary columns and append output to list
        estimates_df_list.append(estimate_df.select(["observations", "pi_estimate"]))

        # Uncomment for un-aggregated graphs
        # Add literal string for coloring all separate simulation scatter points
        # estimate_df = estimate_df.with_columns(pl.lit(sim_title).alias("sim_title"))

        # Keep only necessary columns and append output to list
        # estimates_df_list.append(
        #     estimate_df.select(["observations", "pi_estimate", "sim_title"])
        # )

    # Combine output LazyFrames and groupby Observation index
    # Average Pi estimate values across rows for each simulation "observation" index
    # Sort ascending observation index, collect and convert to pandas for plotting
    multiple_estimate_df = (
        (
            pl.concat(items=estimates_df_list, parallel=True)
            .group_by("observations")
            .mean()
            .sort(by="observations", descending=False)
        )
        .collect()
        .to_pandas()
    )

    # Create a plotly figure if file path is given
    if isinstance(file, str):
        create_estimate_figure(
            file=file,
            df=multiple_estimate_df,
            x_label=x_label,
            y_label=y_label,
        )

    # Uncomment/Comment out related code to make un-aggregated scatter plot
    # Yes I'm lazy and don't want to make a new function for this lol
    # # Do not aggregate and plot with color
    # un_aggregated_df = (
    #     pl.concat(items=estimates_df_list, parallel=True)
    #     .sort(by="observations", descending=False)
    #     .collect()
    #     .to_pandas()
    # )

    # # Create a plotly figure if file path is given
    # if isinstance(file, str):
    #     create_estimate_figure(
    #         file=file,
    #         df=un_aggregated_df,
    #         x_label=x_label,
    #         y_label=y_label,
    #         color="sim_title",
    #     )


if __name__ == "__main__":

    print(f"Square Area: {SQUARE.area}")
    print(f"Inscribed Circle Area: {INSCRIBED_CIRCLE.area}")
    pi_ratio = 4 * (INSCRIBED_CIRCLE.area / SQUARE.area)
    print(f"Area Ratio (Circle Area/Square Area): {pi_ratio}")

    multiple_monte_carlo_example_polars(
        square_polygon=SQUARE,
        inscribed_circle_polygon=INSCRIBED_CIRCLE,
        n_samples=10000,
        n_simulations=50,
        seed=42,
        file=r"src\monte_carlo_example\figs\multiple_monte_carlo_example_polars_50.html",
        x_label="observations",
        y_label="pi_estimate",
    )
