# %%
from pandas import DataFrame
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd

from tools.figures import create_estimate_figure
from tools.generate import (
    INSCRIBED_CIRCLE,
    SQUARE,
    generate_estimates_polars,
)

print(f"Square Area: {SQUARE.area}")
print(f"Inscribed Circle Area: {INSCRIBED_CIRCLE.area}")
pi_ratio = 4 * (INSCRIBED_CIRCLE.area / SQUARE.area)
print(f"Area Ratio (Circle Area/Square Area): {pi_ratio}")

square_polygon = SQUARE
inscribed_circle_polygon = INSCRIBED_CIRCLE
n_samples = 10000
seed = 42
file = None
# file=r"src\monte_carlo_example\figs\monte_carlo_example_polars.html"
x_label = "observations"
y_label = "pi_estimate"

estimate_df = generate_estimates_polars(
    square_polygon=square_polygon,
    inscribed_circle_polygon=inscribed_circle_polygon,
    n_samples=n_samples,
    seed=seed,
    x_label=x_label,
    y_label=y_label,
)

# %%


def plot_monte_carlo_points(
    estimate_df: pd.DataFrame,
    square_polygon: Polygon,
    inscribed_circle_polygon: Polygon,
    observations: int,
) -> None:
    """
    Plot the square, inscribed circle, and Monte Carlo points up to the given observation count.

    Parameters:
        estimate_df (pd.DataFrame): DataFrame containing 'geom' column with Shapely Points.
        square_polygon (Polygon): The bounding square.
        inscribed_circle_polygon (Polygon): The inscribed circle within the square.
        observations (int): Number of points to include in the plot (from the start).
    """

    # Clamp observations to the length of the dataframe to avoid IndexError
    observations = min(observations, len(estimate_df))

    # Filter to the first `observations` rows
    df_subset = (
        estimate_df[estimate_df["observations"] <= observations]
        .copy()
        .reset_index(drop=True)
    )

    # Ensure boolean type
    df_subset["inside"] = df_subset["inside"].astype(bool)

    # Get values from the last record for each subset for printing
    points_inside = df_subset["inside_cumsum"].iloc[-1]
    pi_estimate = df_subset["pi_estimate"].iloc[-1]

    print(f"points inside circle: {points_inside}")
    print(f"total points: {observations}")
    print(f"pi estimate: {pi_estimate} ")

    # Extract x and y coordinates from 'geom' column
    df_subset["x"] = df_subset["geom"].apply(lambda p: p.x)
    df_subset["y"] = df_subset["geom"].apply(lambda p: p.y)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot square
    x_sq, y_sq = square_polygon.exterior.xy
    ax.plot(x_sq, y_sq, color="black", label="Square")

    # Plot circle
    x_circ, y_circ = inscribed_circle_polygon.exterior.xy
    ax.plot(x_circ, y_circ, color="black", label="Inscribed Circle")

    # Plot points
    ax.scatter(
        df_subset.loc[df_subset["inside"], "x"],
        df_subset.loc[df_subset["inside"], "y"],
        color="red",
        s=2,
        label="Inside Circle",
    )
    ax.scatter(
        df_subset.loc[~df_subset["inside"], "x"],
        df_subset.loc[~df_subset["inside"], "y"],
        color="blue",
        s=2,
        label="Outside Circle",
    )

    # Formatting
    ax.set_aspect("equal")
    # No Title, no legend
    # ax.set_title(
    #     f"(points inside circle: {points_inside} / total points: {observations} ) * 4 = {pi_estimate} pi estimate"
    # )
    # ax.legend()
    plt.show()


assert isinstance(estimate_df, pd.DataFrame)
plot_monte_carlo_points(
    estimate_df=estimate_df,
    square_polygon=SQUARE,
    inscribed_circle_polygon=INSCRIBED_CIRCLE,
    observations=1,
)

plot_monte_carlo_points(
    estimate_df=estimate_df,
    square_polygon=SQUARE,
    inscribed_circle_polygon=INSCRIBED_CIRCLE,
    observations=10,
)

plot_monte_carlo_points(
    estimate_df=estimate_df,
    square_polygon=SQUARE,
    inscribed_circle_polygon=INSCRIBED_CIRCLE,
    observations=100,
)

plot_monte_carlo_points(
    estimate_df=estimate_df,
    square_polygon=SQUARE,
    inscribed_circle_polygon=INSCRIBED_CIRCLE,
    observations=1000,
)

plot_monte_carlo_points(
    estimate_df=estimate_df,
    square_polygon=SQUARE,
    inscribed_circle_polygon=INSCRIBED_CIRCLE,
    observations=10000,
)
# %%


# Create a plotly figure if file path is given
if isinstance(file, str):
    assert isinstance(estimate_df, DataFrame)  # Satisfy pyright type checker
    create_estimate_figure(
        file=file,
        df=estimate_df,
        x_label=x_label,
        y_label=y_label,
    )


# if __name__ == "__main__":
