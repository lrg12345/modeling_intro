import matplotlib.pyplot as plt
import numpy as np


def generate_triangle(width=800, height=700, border=100):
    """
    Generate and display an RGB research triangle.

    The three vertices represent:
        Red   = Analytical
        Green = Physical
        Blue  = Data-Driven

    Parameters
    ----------
    width : int, optional
        Image width in pixels. Default is 800.
    height : int, optional
        Image height in pixels. Default is 700.
    border : int, optional
        Distance between the triangle vertices and image edges.
        Default is 100.

    Returns
    -------
    analytical_vertex : numpy.ndarray
        Coordinates of the analytical vertex.
    physical_vertex : numpy.ndarray
        Coordinates of the physical vertex.
    data_vertex : numpy.ndarray
        Coordinates of the data-driven vertex.
    """

    image = np.ones((height, width, 3), dtype=float)

    analytical_vertex = np.array([width / 2, border])
    physical_vertex = np.array([border, height - border])
    data_vertex = np.array([width - border, height - border])

    maximum_distance = max(
        np.linalg.norm(analytical_vertex - physical_vertex),
        np.linalg.norm(analytical_vertex - data_vertex),
        np.linalg.norm(physical_vertex - data_vertex),
    )

    for y_coordinate in range(height):
        for x_coordinate in range(width):
            pixel_point = np.array([x_coordinate, y_coordinate])

            vector_0 = data_vertex - analytical_vertex
            vector_1 = physical_vertex - analytical_vertex
            vector_2 = pixel_point - analytical_vertex

            dot_00 = np.dot(vector_0, vector_0)
            dot_01 = np.dot(vector_0, vector_1)
            dot_02 = np.dot(vector_0, vector_2)
            dot_11 = np.dot(vector_1, vector_1)
            dot_12 = np.dot(vector_1, vector_2)

            denominator = dot_00 * dot_11 - dot_01 * dot_01

            barycentric_u = (dot_11 * dot_02 - dot_01 * dot_12) / denominator

            barycentric_v = (dot_00 * dot_12 - dot_01 * dot_02) / denominator

            point_is_inside = (
                barycentric_u >= 0
                and barycentric_v >= 0
                and barycentric_u + barycentric_v <= 1
            )

            if point_is_inside:
                distance_to_analytical = np.linalg.norm(pixel_point - analytical_vertex)

                distance_to_physical = np.linalg.norm(pixel_point - physical_vertex)

                distance_to_data = np.linalg.norm(pixel_point - data_vertex)

                red_intensity = 1.0 - distance_to_analytical / maximum_distance

                green_intensity = 1.0 - distance_to_physical / maximum_distance

                blue_intensity = 1.0 - distance_to_data / maximum_distance

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

    return analytical_vertex, physical_vertex, data_vertex


def plot_point(
    analytical_vertex,
    physical_vertex,
    data_vertex,
    analytical_weight=1 / 3,
    physical_weight=1 / 3,
    data_weight=1 / 3,
    point_color="black",
    point_size=120,
    point_marker="o",
    project_label="My Project",
):
    """
    Plot a research project point inside the triangle.

    The analytical, physical, and data-driven weights must be
    nonnegative and add up to 1.
    """

    total_weight = analytical_weight + physical_weight + data_weight

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "Analytical, physical, and data-driven weights must add up to 1."
        )

    if analytical_weight < 0 or physical_weight < 0 or data_weight < 0:
        raise ValueError("Project weights cannot be negative.")

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

    plt.show()


if __name__ == "__main__":
    import sys

    output_file = "triangle.png"
    analytical_weight = 1 / 3
    physical_weight = 1 / 3
    data_weight = 1 / 3

    if len(sys.argv) >= 2:
        output_file = sys.argv[1]

    if len(sys.argv) == 5:
        analytical_weight = float(sys.argv[2])
        physical_weight = float(sys.argv[3])
        data_weight = float(sys.argv[4])

    analytical, physical, data = generate_triangle()

    plot_point(
        analytical,
        physical,
        data,
        analytical_weight=analytical_weight,
        physical_weight=physical_weight,
        data_weight=data_weight,
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
