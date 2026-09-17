"""
Create and plot points on an RGB research triangle.

The triangle represents three research dimensions:

    Red   = Analytical
    Green = Physical
    Blue  = Data-Driven

The two main functions are:

    generate_triangle()
        Creates the RGB triangle.

    plot_point()
        Adds a research project to the triangle using three weights.

The module can also be run directly from the command line.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


__all__ = ["generate_triangle", "plot_point"]


def _is_point_inside_triangle(point, vertex_a, vertex_b, vertex_c):
    """Return True if a point lies inside a triangle."""

    vector_0 = vertex_c - vertex_a
    vector_1 = vertex_b - vertex_a
    vector_2 = point - vertex_a

    dot_00 = np.dot(vector_0, vector_0)
    dot_01 = np.dot(vector_0, vector_1)
    dot_02 = np.dot(vector_0, vector_2)
    dot_11 = np.dot(vector_1, vector_1)
    dot_12 = np.dot(vector_1, vector_2)

    denominator = dot_00 * dot_11 - dot_01 * dot_01

    barycentric_u = (
        dot_11 * dot_02 - dot_01 * dot_12
    ) / denominator

    barycentric_v = (
        dot_00 * dot_12 - dot_01 * dot_02
    ) / denominator

    return (
        barycentric_u >= 0
        and barycentric_v >= 0
        and barycentric_u + barycentric_v <= 1
    )


def generate_triangle(width=800, height=700, border=100):
    """
    Generate and display an RGB research triangle.

    The triangle represents three research dimensions:

        Analytical  = red
        Physical    = green
        Data-Driven = blue

    Parameters
    ----------
    width : int, optional
        Image width in pixels. Default is 800.
    height : int, optional
        Image height in pixels. Default is 700.
    border : int, optional
        Distance between the triangle and image edges.
        Default is 100.

    Returns
    -------
    dict
        Dictionary containing the coordinates of the three vertices.
        The keys are "analytical", "physical", and "data".

    Examples
    --------
    >>> triangle = generate_triangle()
    """

    image = np.ones((height, width, 3), dtype=float)

    vertices = {
        "analytical": np.array([width / 2, border]),
        "physical": np.array([border, height - border]),
        "data": np.array([width - border, height - border]),
    }

    analytical_vertex = vertices["analytical"]
    physical_vertex = vertices["physical"]
    data_vertex = vertices["data"]

    maximum_distance = max(
        np.linalg.norm(analytical_vertex - physical_vertex),
        np.linalg.norm(analytical_vertex - data_vertex),
        np.linalg.norm(physical_vertex - data_vertex),
    )

    for y_coordinate in range(height):
        for x_coordinate in range(width):
            pixel_point = np.array([x_coordinate, y_coordinate])

            if _is_point_inside_triangle(
                pixel_point,
                analytical_vertex,
                physical_vertex,
                data_vertex,
            ):
                distance_to_analytical = np.linalg.norm(
                    pixel_point - analytical_vertex
                )

                distance_to_physical = np.linalg.norm(
                    pixel_point - physical_vertex
                )

                distance_to_data = np.linalg.norm(
                    pixel_point - data_vertex
                )

                red_intensity = (
                    1.0
                    - distance_to_analytical / maximum_distance
                )

                green_intensity = (
                    1.0
                    - distance_to_physical / maximum_distance
                )

                blue_intensity = (
                    1.0
                    - distance_to_data / maximum_distance
                )

                pixel_color = np.array(
                    [
                        red_intensity,
                        green_intensity,
                        blue_intensity,
                    ]
                )

                pixel_color /= pixel_color.max()

                image[y_coordinate, x_coordinate] = pixel_color

    plt.figure(figsize=(8, 7))
    plt.imshow(image)

    plt.text(
        analytical_vertex[0],
        analytical_vertex[1] - 25,
        "Analytical",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.text(
        physical_vertex[0],
        physical_vertex[1] + 35,
        "Physical",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.text(
        data_vertex[0],
        data_vertex[1] + 35,
        "Data-Driven",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.axis("off")

    return vertices


def plot_point(
    triangle,
    analytical_weight=1 / 3,
    physical_weight=1 / 3,
    data_weight=1 / 3,
    point_color="black",
    point_size=120,
    point_marker="o",
    project_label="My Project",
    save_file=None,
    show=True,
):
    """
    Plot a research project inside an RGB research triangle.

    The location of the point is determined by analytical,
    physical, and data-driven weights. The three weights must
    be nonnegative and add up to 1.

    Parameters
    ----------
    triangle : dict
        Triangle returned by generate_triangle().
    analytical_weight : float, optional
        Analytical contribution. Default is 1/3.
    physical_weight : float, optional
        Physical contribution. Default is 1/3.
    data_weight : float, optional
        Data-driven contribution. Default is 1/3.
    point_color : str, optional
        Color of the plotted point. Default is "black".
    point_size : int, optional
        Size of the plotted point. Default is 120.
    point_marker : str, optional
        Matplotlib marker style. Default is "o".
    project_label : str, optional
        Label displayed next to the point.
        Default is "My Project".
    save_file : str or None, optional
        File path for saving the figure. If None, the figure
        is not saved.
    show : bool, optional
        If True, display the figure. Default is True.

    Returns
    -------
    numpy.ndarray
        The [x, y] coordinates of the plotted project point.

    Examples
    --------
    >>> triangle = generate_triangle()
    >>> plot_point(
    ...     triangle,
    ...     analytical_weight=0.5,
    ...     physical_weight=0.3,
    ...     data_weight=0.2,
    ... )
    """

    total_weight = (
        analytical_weight
        + physical_weight
        + data_weight
    )

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "Analytical, physical, and data-driven weights "
            "must add up to 1."
        )

    if (
        analytical_weight < 0
        or physical_weight < 0
        or data_weight < 0
    ):
        raise ValueError(
            "Project weights cannot be negative."
        )

    analytical_vertex = triangle["analytical"]
    physical_vertex = triangle["physical"]
    data_vertex = triangle["data"]

    project_position = (
        analytical_weight * analytical_vertex
        + physical_weight * physical_vertex
        + data_weight * data_vertex
    )

    plt.scatter(
        project_position[0],
        project_position[1],
        color=point_color,
        s=point_size,
        marker=point_marker,
        edgecolor="white",
        linewidth=2,
        zorder=5,
    )

    plt.text(
        project_position[0],
        project_position[1] - 20,
        project_label,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    if save_file is not None:
        plt.savefig(
            save_file,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return project_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate an RGB research triangle and plot "
            "a weighted research project point."
        )
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="triangle.png",
        help="Output image filename. Default: triangle.png",
    )

    parser.add_argument(
        "analytical",
        nargs="?",
        type=float,
        default=1 / 3,
        help="Analytical weight. Default: 1/3",
    )

    parser.add_argument(
        "physical",
        nargs="?",
        type=float,
        default=1 / 3,
        help="Physical weight. Default: 1/3",
    )

    parser.add_argument(
        "data",
        nargs="?",
        type=float,
        default=1 / 3,
        help="Data-driven weight. Default: 1/3",
    )

    args = parser.parse_args()

    triangle = generate_triangle()

    plot_point(
        triangle,
        analytical_weight=args.analytical,
        physical_weight=args.physical,
        data_weight=args.data,
        save_file=args.output,
        show=False,
    )

    print(f"Triangle saved to {args.output}")