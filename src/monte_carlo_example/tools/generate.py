import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import Point, Polygon
from shapely import contains

# Constant shapes
SQUARE = Polygon(((0, 0), (0, 4), (4, 4), (4, 0)))

# From the .buffer() docstring on `quad_segs`
# ```
# 16-gon approx of a unit radius circle:
# >>> g.buffer(1.0).area
# 3.1365484905459398

# 128-gon approximation:
# >>> g.buffer(1.0, 128).area
# 3.1415138011443013

# triangle approximation:
# >>> g.buffer(1.0, 3).area
# 3.0
# ```
INSCRIBED_CIRCLE = SQUARE.centroid.buffer(
    distance=2,  # Radius
    quad_segs=1000,  # Such circular, much wow
)


def generate_random_points(
    square_polygon: Polygon, n_points: int, seed: int | None = None
) -> list[Point]:
    """
    Generate a list of random points within the bounding box of a given polygon.

    This function samples `n_points` uniformly distributed random points within
    the axis-aligned bounding box of the input `square_polygon`. If `square_polygon` is
    not a square the points are not guaranteed to lie inside the polygon, only within its
    bounding box.

    Parameters:
        square_polygon (Polygon): A Shapely Polygon whose bounding box is used for sampling.
        n_points (int): The number of random points to generate.
        seed (int | None, optional): Optional seed for reproducibility. Defaults to None.

    Returns:
        list[Point]: A list of Shapely Point objects representing the sampled coordinates.
    """

    # Localized rng with own seed
    rng = np.random.default_rng(seed=seed)

    # Get polygon boundaries
    minx, miny, maxx, maxy = square_polygon.bounds

    # Generate random coordinates within the bounding box
    random_x = rng.uniform(minx, maxx, n_points)
    random_y = rng.uniform(miny, maxy, n_points)

    # Construct list of points from xy coordinate arrays
    sampled_points = [Point(x, y) for x, y in zip(random_x, random_y)]

    return sampled_points


def generate_estimates_for_loop(
    square_polygon: Polygon,
    inscribed_circle_polygon: Polygon,
    n_samples: int,
    seed: int | None = None,
    x_label: str = "observations",
    y_label: str = "pi_estimate",
) -> pd.DataFrame:
    """
    Estimate the value of `pi` using the Monte Carlo method by sampling random points
    within a square and checking how many fall inside an inscribed circle.

    This function performs a step-by-step estimation of `pi` by:
        1. Generating `n_samples` random points within the bounding box of a square polygon.
        2. Counting how many of those points fall within the inscribed circle.
        3. Computing the running estimate of `pi` at each step as: `pi` ≈ 4 * (inside / total).

    Parameters:
        square_polygon (Polygon): A Shapely Polygon whose bounding box is used for sampling. Assumed to be a true square.
        inscribed_circle_polygon (Polygon): A Shapely Polygon representing the circle inscribed within the square.
        n_samples (int): Number of random points to generate for the simulation.
        seed (int | None, optional): Optional seed for reproducibility. Defaults to None.
        x_label (str, optional): Label for the x-axis (e.g., number of observations). Defaults to "observations".
        y_label (str, optional): Label for the y-axis (e.g., running estimate of `pi`). Defaults to "pi_estimate".

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - `x_label`: Cumulative number of samples (1 to `n_samples`)
            - `y_label`: Corresponding running estimates of `pi`
    """

    # Initialize variables
    observation_count = 0
    inside_circle = 0
    estimates = []
    observations = []

    # Get a list of random points inside input square polygon
    points = generate_random_points(
        square_polygon=square_polygon, n_points=n_samples, seed=seed
    )

    # For each random point
    for pt in points:

        # Increment count
        observation_count += 1

        # if intersects(random_point, inscribed_circle):
        #     inside_circle += 1
        if contains(inscribed_circle_polygon, pt):
            inside_circle += 1

        # Estimate pi at each step in the loop: 4 * (inside / total)
        estimation = 4 * (inside_circle / observation_count)

        # Add current values to the accumulating lists
        estimates.append(estimation)
        observations.append(observation_count)

    # Initialize as dataframe for output/figures
    df = pd.DataFrame({x_label: observations, y_label: estimates})

    return df


def generate_estimates_polars(
    square_polygon: Polygon,
    inscribed_circle_polygon: Polygon,
    n_samples: int,
    seed: int | None = None,
    x_label: str = "observations",
    y_label: str = "pi_estimate",
    return_lazy_frame: bool = False,
) -> pd.DataFrame | pl.LazyFrame:
    """
    Estimate the value of `pi` using the Monte Carlo method, implemented with a Polars LazyFrame pipeline for performance.

    This function performs a step-by-step estimation of `pi` by:
        1. Generating `n_samples` random points within the bounding box of a square polygon.
        2. Counting how many of those points fall within the inscribed circle.
        3. Computing the running estimate of `pi` at each step as: `pi` ≈ 4 * (inside / total).

    The estimation is executed efficiently using Polars lazy evaluation and
    vectorized operations.

    Parameters:
        square_polygon (Polygon): A Shapely Polygon whose bounding box is used for sampling. Assumed to be a true square.
        inscribed_circle_polygon (Polygon): A Shapely Polygon representing the circle inscribed within the square.
        n_samples (int): Number of random points to generate for the simulation.
        seed (int | None, optional): Optional seed for reproducibility. Defaults to None.
        x_label (str, optional): Label for the x-axis (e.g., number of observations). Defaults to "observations".
        y_label (str, optional): Label for the y-axis (e.g., running estimate of `pi`). Defaults to "pi_estimate".
        return_lazy_frame (bool, optional): If True, returns the uncollected Polars LazyFrame instead of a collected
            Pandas DataFrame. Defaults to False.

    Returns:
        pd.DataFrame | pl.LazyFrame: A Pandas DataFrame with the cumulative `pi` estimates,
        or a Polars LazyFrame if `return_lazy_frame=True`. The returned frame includes:
            - `x_label`: Cumulative number of samples (1 to `n_samples`)
            - `y_label`: Corresponding running estimates of `pi`
    """

    # Create LazyFrame from iterable of random points inside input square polygon
    df = pl.LazyFrame(
        data={
            "geom": generate_random_points(
                square_polygon=square_polygon, n_points=n_samples, seed=seed
            )
        }
    )

    # Broadcast Polygon contains Point calculation across the geometry column
    df = df.with_columns(
        pl.col("geom")
        .map_elements(
            function=(lambda pt: contains(inscribed_circle_polygon, pt)),
            return_dtype=pl.Boolean,
        )
        .cast(pl.Int8)
        .alias("inside")
    )

    # Compute cumulative count of inside points
    # Get increasing tally of total observations made
    df = df.with_columns(
        [
            pl.col("inside").cum_sum().alias("inside_cumsum"),
            pl.arange(1, pl.len() + 1).alias(x_label),
        ]
    )

    # Estimate pi at each cumulative step: 4 * (inside / total)
    df = df.with_columns(
        (4 * (pl.col("inside_cumsum") / pl.col(x_label))).alias(y_label)
    )

    # Optionally return uncollected LazyFrame
    if return_lazy_frame:
        return df
    else:
        return df.collect().to_pandas()
