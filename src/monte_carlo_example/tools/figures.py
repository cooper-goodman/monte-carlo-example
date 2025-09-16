import pandas as pd
import plotly.express as px


def create_estimate_figure(
    file: str,
    df: pd.DataFrame,
    x_label: str = "observations",
    y_label: str = "pi_estimate",
    color: str | None = None,
) -> None:
    """
    Create a scatter plot of `pi` estimates and save it as an HTML file.

    The plot includes a horizontal reference line for the true value of `pi`.

    Parameters:
        file (str): Path to the output HTML file.
        df (pd.DataFrame): DataFrame containing the data to plot.
        x_label (str, optional): Label for the x-axis (e.g., number of observations). Defaults to "observations".
        y_label (str, optional): Label for the y-axis (e.g., running estimate of `pi`). Defaults to "pi_estimate".
        color (str | None, optional) Dimension for scatter dot colors. Defaults to None.

    Example:

        ```python

        estimate_df = pd.DataFrame({"observations": [1, 2, 3], "pi_estimate": [3, 3.1, 3.14]})

        create_estimate_figure(
                file="C:\\Users\\Archimedes\\Downloads\\monte_carlo_example.html",
                df=estimate_df,
                x_label="observations",
                y_label="pi_estimate",
            )
        ```
    """

    fig = px.scatter(data_frame=df, x=x_label, y=y_label, color=color)
    fig.add_hline(y=3.14159265359, line_color="red")
    fig.write_html(file)
