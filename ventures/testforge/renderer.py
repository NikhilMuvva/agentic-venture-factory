from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc as ArcPatch, Circle, Polygon as PolygonPatch

from models import (
    AngleMarker,
    Arc,
    BarChartVisual,
    BoxPlotVisual,
    CirclePrimitive,
    CircleVisual,
    DotPlotVisual,
    GeometryPrimitive,
    GeometryPoint,
    GeometryVisual,
    GraphSeries,
    HistogramVisual,
    Label,
    Polygon,
    RightAngleMarker,
    ScatterPlotVisual,
    Segment,
    Solid3DVisual,
    TableVisual,
    VisualSpec,
    XYGraphVisual,
)


FIGURE_SIZE = (7.2, 4.6)
DPI = 150


def _finish(figure, path: Path, title: str | None = None, to_scale: bool = True) -> Path:
    if title:
        figure.suptitle(title, fontsize=12)
    if not to_scale:
        figure.text(0.5, 0.015, "Note: Figure not drawn to scale.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04 if not to_scale else 0, 1, 0.95 if title else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, format="png", bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_graph_series(axis, series: GraphSeries, x_values: np.ndarray) -> None:
    if series.kind in ("line", "inequality_boundary"):
        y_values = series.slope * x_values + series.intercept
        linestyle = "--" if series.kind == "inequality_boundary" and series.relation in ("<", ">") else "-"
        axis.plot(x_values, y_values, label=series.label or "Line", linestyle=linestyle)
        if series.kind == "inequality_boundary" and series.relation:
            if series.relation in ("<", "<="):
                axis.fill_between(x_values, y_values, axis.get_ylim()[0], alpha=0.16)
            else:
                axis.fill_between(x_values, y_values, axis.get_ylim()[1], alpha=0.16)
    elif series.kind == "quadratic":
        y_values = series.a * x_values ** 2 + series.b * x_values + series.c
        axis.plot(x_values, y_values, label=series.label or "Quadratic")
    elif series.kind == "exponential":
        y_values = series.initial_value * np.power(series.base, x_values)
        axis.plot(x_values, y_values, label=series.label or "Exponential")
    elif series.kind == "points":
        point_x, point_y = zip(*series.points)
        axis.scatter(point_x, point_y, label=series.label or "Points", zorder=3)


def _draw_xy_graph(spec: XYGraphVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    x_values = np.linspace(spec.x_min, spec.x_max, 500)
    axis.set_xlim(spec.x_min, spec.x_max)
    axis.set_ylim(spec.y_min, spec.y_max)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    for series in spec.series:
        _plot_graph_series(axis, series, x_values)
    axis.set_xlabel(spec.x_label or "x")
    axis.set_ylabel(spec.y_label or "y")
    axis.grid(spec.grid, alpha=0.3)
    if any(series.label for series in spec.series):
        axis.legend(loc="best", fontsize=8)
    return _finish(figure, path)


def _draw_table(spec: TableVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axis("off")
    table = axis.table(
        cellText=spec.rows,
        colLabels=spec.column_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    return _finish(figure, path, spec.title)


def _draw_scatterplot(spec: ScatterPlotVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    x_values, y_values = zip(*spec.points)
    axis.scatter(x_values, y_values, color="#176B87", s=42, zorder=3)
    if spec.best_fit_line:
        low, high = min(x_values), max(x_values)
        line_x = np.linspace(low, high, 100)
        line_y = spec.best_fit_line.slope * line_x + spec.best_fit_line.intercept
        axis.plot(line_x, line_y, color="#D1495B", linewidth=2, label="Line of best fit")
        axis.legend()
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel(spec.y_label)
    axis.grid(True, alpha=0.25)
    return _finish(figure, path)


def _draw_bar_chart(spec: BarChartVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.bar(spec.labels, spec.values, color="#2A9D8F")
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel(spec.y_label)
    axis.grid(axis="y", alpha=0.25)
    return _finish(figure, path)


def _draw_histogram(spec: HistogramVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.bar(
        spec.bin_edges[:-1],
        spec.frequencies,
        width=np.diff(spec.bin_edges),
        align="edge",
        color="#E9C46A",
        edgecolor="#333333",
    )
    axis.set_xlabel(spec.x_label)
    axis.set_ylabel(spec.y_label)
    axis.set_xticks(spec.bin_edges)
    axis.grid(axis="y", alpha=0.25)
    return _finish(figure, path)


def _draw_dot_plot(spec: DotPlotVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    if spec.values:
        values, frequencies = np.unique(spec.values, return_counts=True)
    else:
        values = np.array([item.value for item in spec.value_frequencies])
        frequencies = np.array([item.frequency for item in spec.value_frequencies])
    for value, frequency in zip(values, frequencies):
        y_values = np.arange(1, int(frequency) + 1)
        axis.scatter([value] * len(y_values), y_values, color="#176B87", s=36)
    axis.set_xlabel(spec.x_label)
    axis.set_yticks([])
    axis.set_ylim(0, max(frequencies) + 1)
    axis.grid(axis="x", alpha=0.25)
    return _finish(figure, path)


def _draw_box_plot(spec: BoxPlotVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.bxp([{
        "label": "Distribution",
        "whislo": spec.min,
        "q1": spec.q1,
        "med": spec.median,
        "q3": spec.q3,
        "whishi": spec.max,
        "fliers": [],
    }], showfliers=False, vert=False)
    axis.set_xlabel(spec.x_label)
    axis.set_yticks([])
    axis.grid(axis="x", alpha=0.25)
    return _finish(figure, path)


def _collect_geometry_points(spec: GeometryVisual | CircleVisual) -> list[tuple[float, float]]:
    points = []
    if isinstance(spec, GeometryVisual):
        for item in spec.primitives:
            if item.kind == "point":
                points.append((item.x, item.y))
            elif item.kind == "segment":
                points.extend((item.start, item.end))
            elif item.kind == "polygon":
                points.extend(item.vertices)
            elif item.kind in ("circle", "arc"):
                x, y = item.center
                points.extend(((x - item.radius, y - item.radius), (x + item.radius, y + item.radius)))
            elif item.kind == "angle":
                points.extend((item.vertex, item.ray1_point, item.ray2_point))
            elif item.kind == "right_angle":
                points.extend((item.vertex, item.along_first_ray, item.along_second_ray))
            elif item.kind == "label":
                points.append(item.position)
    else:
        cx, cy = spec.center
        points.extend(((cx - spec.radius, cy - spec.radius), (cx + spec.radius, cy + spec.radius)))
        points.extend((point.x, point.y) for point in spec.points)
        for segment in [*spec.radii, *spec.chords, *spec.tangents]:
            points.extend((segment.start, segment.end))
    return points or [(-1, -1), (1, 1)]


def _draw_angle(axis, vertex, first, second, label: str | None = None) -> None:
    start = math.degrees(math.atan2(first[1] - vertex[1], first[0] - vertex[0]))
    end = math.degrees(math.atan2(second[1] - vertex[1], second[0] - vertex[0]))
    axis.add_patch(ArcPatch(vertex, 0.6, 0.6, theta1=start, theta2=end, color="#D1495B", linewidth=1.4))
    if label:
        axis.text(vertex[0] + 0.38, vertex[1] + 0.28, label, fontsize=9)


def _draw_segment(axis, segment: Segment) -> None:
    axis.plot([segment.start[0], segment.end[0]], [segment.start[1], segment.end[1]], color="#176B87", linewidth=1.8)
    if segment.label:
        midpoint = ((segment.start[0] + segment.end[0]) / 2, (segment.start[1] + segment.end[1]) / 2)
        axis.text(*midpoint, segment.label, fontsize=9, backgroundcolor="white")


def _draw_geometry(spec: GeometryVisual | CircleVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    all_points = _collect_geometry_points(spec)
    xs, ys = zip(*all_points)
    x_span = max(max(xs) - min(xs), 1.0)
    y_span = max(max(ys) - min(ys), 1.0)
    margin = 0.12 * max(x_span, y_span)
    axis.set_xlim(min(xs) - margin, max(xs) + margin)
    axis.set_ylim(min(ys) - margin, max(ys) + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    if isinstance(spec, CircleVisual):
        axis.add_patch(Circle(spec.center, spec.radius, fill=False, linewidth=2, color="#176B87"))
        for segment in [*spec.radii, *spec.chords, *spec.tangents]:
            _draw_segment(axis, segment)
        for arc in spec.arcs:
            axis.add_patch(ArcPatch(arc.center, 2 * arc.radius, 2 * arc.radius,
                                    theta1=arc.start_degrees, theta2=arc.end_degrees,
                                    color="#D1495B", linewidth=2))
            if arc.label:
                angle = math.radians((arc.start_degrees + arc.end_degrees) / 2)
                axis.text(arc.center[0] + 1.08 * arc.radius * math.cos(angle),
                          arc.center[1] + 1.08 * arc.radius * math.sin(angle), arc.label, fontsize=9)
        for marker in spec.angle_markers:
            _draw_angle(axis, marker.vertex, marker.ray1_point, marker.ray2_point, marker.label)
        for point in spec.points:
            axis.scatter([point.x], [point.y], color="#D1495B", zorder=4)
            axis.text(point.x, point.y, f" {point.name}", fontsize=9)
        if spec.equation:
            axis.text(0.02, 0.98, spec.equation, transform=axis.transAxes, va="top", fontsize=9)
        return _finish(figure, path, to_scale=spec.to_scale)

    for item in spec.primitives:
        if item.kind == "point":
            axis.scatter([item.x], [item.y], color="#D1495B", zorder=4)
            if item.name:
                axis.annotate(item.name, (item.x, item.y), xytext=(4, 4), textcoords="offset points", fontsize=9)
        elif item.kind == "segment":
            segment = Segment(start=item.start, end=item.end, label=item.label)
            _draw_segment(axis, segment)
        elif item.kind == "polygon":
            axis.add_patch(PolygonPatch(item.vertices, closed=True, fill=False, linewidth=2, color="#176B87"))
            if item.label:
                center_x = sum(point[0] for point in item.vertices) / len(item.vertices)
                center_y = sum(point[1] for point in item.vertices) / len(item.vertices)
                axis.text(center_x, center_y, item.label, fontsize=9)
        elif item.kind == "circle":
            axis.add_patch(Circle(item.center, item.radius, fill=False, linewidth=2, color="#176B87"))
            if item.label:
                axis.text(item.center[0], item.center[1], item.label, fontsize=9)
        elif item.kind == "arc":
            axis.add_patch(ArcPatch(item.center, 2 * item.radius, 2 * item.radius,
                                    theta1=item.start_degrees, theta2=item.end_degrees,
                                    color="#D1495B", linewidth=2))
            if item.label:
                axis.text(*item.center, item.label, fontsize=9)
        elif item.kind == "angle":
            _draw_angle(axis, item.vertex, item.ray1_point, item.ray2_point, item.label)
        elif item.kind == "label":
            axis.annotate(item.text, item.position, fontsize=9)
        elif item.kind == "right_angle":
            vertex = np.array(item.vertex)
            ray_a = np.array(item.along_first_ray) - vertex
            ray_b = np.array(item.along_second_ray) - vertex
            ray_a = ray_a / np.linalg.norm(ray_a) * item.size
            ray_b = ray_b / np.linalg.norm(ray_b) * item.size
            corners = [vertex + ray_a, vertex + ray_a + ray_b, vertex + ray_b]
            axis.plot([point[0] for point in corners], [point[1] for point in corners], color="#D1495B")
    return _finish(figure, path, to_scale=spec.to_scale)


def _draw_solid(spec: Solid3DVisual, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axis("off")
    axis.set_aspect("equal")
    if spec.solid_type == "rectangular_prism":
        assert spec.width is not None and spec.depth is not None and spec.height is not None
        width, depth, height = spec.width, spec.depth, spec.height
        dx, dy = depth * 0.35, depth * 0.22
        front = [(0, 0), (width, 0), (width, height), (0, height)]
        back = [(x + dx, y + dy) for x, y in front]
        for points in (front, back):
            axis.add_patch(PolygonPatch(points, fill=False, linewidth=1.8))
        for first, second in zip(front, back):
            axis.plot([first[0], second[0]], [first[1], second[1]], color="#176B87")
        axis.text(width / 2, -0.35, f"width = {width:g}", ha="center")
        axis.text(width + dx + 0.1, height / 2 + dy, f"height = {height:g}", rotation=90)
    elif spec.solid_type in ("cylinder", "cone"):
        assert spec.radius is not None and spec.height is not None
        radius, height = spec.radius, spec.height
        theta = np.linspace(0, 2 * np.pi, 200)
        axis.plot(radius * np.cos(theta), 0.18 * radius * np.sin(theta), color="#176B87")
        axis.plot(radius * np.cos(theta), height + 0.18 * radius * np.sin(theta), color="#176B87")
        axis.plot([-radius, -radius], [0, height], color="#176B87")
        axis.plot([radius, radius], [0, height], color="#176B87")
        if spec.solid_type == "cone":
            axis.plot([-radius, 0, radius], [0, height, 0], color="#176B87")
        else:
            axis.plot([-radius, radius], [height, height], color="#176B87")
    elif spec.solid_type == "pyramid":
        assert spec.width is not None and spec.depth is not None and spec.height is not None
        width, depth, height = spec.width, spec.depth, spec.height
        base = [(0, 0), (width, 0), (width + 0.6, 0.5), (0.6, 0.5)]
        apex = (width / 2 + 0.3, height)
        axis.add_patch(PolygonPatch(base, fill=False, linewidth=1.8))
        for point in base:
            axis.plot([apex[0], point[0]], [apex[1], point[1]], color="#176B87")
    else:
        assert spec.radius is not None
        sphere = Circle((0, 0), spec.radius, fill=False, linewidth=1.8, color="#176B87")
        axis.add_patch(sphere)
        axis.add_patch(ArcPatch((0, 0), 2 * spec.radius, 0.8 * spec.radius, theta1=0, theta2=360, color="#D1495B"))
    axis.margins(0.25)
    return _finish(figure, path, spec.label, to_scale=False)


def render_visual(spec: VisualSpec, output_path: str | Path) -> Path:
    """Render validated structured visual data to a deterministic PNG."""
    path = Path(output_path)
    if isinstance(spec, TableVisual):
        return _draw_table(spec, path)
    if isinstance(spec, XYGraphVisual):
        return _draw_xy_graph(spec, path)
    if isinstance(spec, ScatterPlotVisual):
        return _draw_scatterplot(spec, path)
    if isinstance(spec, BarChartVisual):
        return _draw_bar_chart(spec, path)
    if isinstance(spec, HistogramVisual):
        return _draw_histogram(spec, path)
    if isinstance(spec, DotPlotVisual):
        return _draw_dot_plot(spec, path)
    if isinstance(spec, BoxPlotVisual):
        return _draw_box_plot(spec, path)
    if isinstance(spec, (GeometryVisual, CircleVisual)):
        return _draw_geometry(spec, path)
    if isinstance(spec, Solid3DVisual):
        return _draw_solid(spec, path)
    raise TypeError(f"Unsupported visual specification: {type(spec).__name__}")


def student_visible_visual_description(spec: VisualSpec) -> str:
    """Describe only rendered visual content, with no answer/explanation fields."""
    if isinstance(spec, TableVisual):
        rows = [" | ".join(map(str, spec.column_headers))]
        rows.extend(" | ".join(map(str, row)) for row in spec.rows)
        return "Table:\n" + "\n".join(rows)
    if isinstance(spec, XYGraphVisual):
        summaries = []
        for series in spec.series:
            if series.kind in ("line", "inequality_boundary"):
                summaries.append(f"line y={series.slope}x+{series.intercept}")
            elif series.kind == "quadratic":
                summaries.append(f"quadratic y={series.a}x^2+{series.b}x+{series.c}")
            elif series.kind == "exponential":
                summaries.append(f"exponential y={series.initial_value}({series.base})^x")
            else:
                summaries.append(f"plotted points {series.points}")
        return f"Coordinate graph x=[{spec.x_min}, {spec.x_max}], y=[{spec.y_min}, {spec.y_max}]: " + "; ".join(summaries)
    if isinstance(spec, ScatterPlotVisual):
        return f"Scatterplot points {spec.points}; axes {spec.x_label} and {spec.y_label}. Best-fit line: {spec.best_fit_line}."
    if isinstance(spec, DotPlotVisual):
        return f"Dot plot values={spec.values}; frequencies={spec.value_frequencies}."
    if isinstance(spec, BoxPlotVisual):
        return f"Box plot five-number summary: {spec.min}, {spec.q1}, {spec.median}, {spec.q3}, {spec.max}."
    if isinstance(spec, HistogramVisual):
        return f"Histogram bin edges={spec.bin_edges}; frequencies={spec.frequencies}."
    if isinstance(spec, BarChartVisual):
        return f"Bar chart labels={spec.labels}; values={spec.values}."
    return f"Mathematical diagram spec: {spec.model_dump_json(exclude={'visual_type'})}"
