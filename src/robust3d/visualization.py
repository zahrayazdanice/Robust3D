"""Plotly and Matplotlib visualizations generated from computed geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "reference": "#6C8BFF",
    "input": "#F59E72",
    "poisson": "#55D6BE",
    "bpa": "#B68CFF",
    "alpha": "#F2C14E",
}


def _scene_style() -> dict:
    axis = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    return dict(
        xaxis=axis,
        yaxis=axis,
        zaxis=axis,
        aspectmode="data",
        camera=dict(eye=dict(x=1.35, y=1.25, z=0.9), up=dict(x=0, y=0, z=1)),
        bgcolor="#0A0F1C",
    )


def mesh_trace(
    mesh: o3d.geometry.TriangleMesh,
    name: str,
    color: str,
    scene: str = "scene",
    max_triangles: int = 35000,
) -> go.Mesh3d:
    view = mesh
    if len(mesh.triangles) > max_triangles:
        try:
            view = mesh.simplify_quadric_decimation(max_triangles)
        except RuntimeError:
            view = mesh
    vertices = np.asarray(view.vertices)
    triangles = np.asarray(view.triangles)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        name=name, color=color, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.38, diffuse=0.72, specular=0.38, roughness=0.65, fresnel=0.12),
        lightposition=dict(x=100, y=80, z=160), hoverinfo="skip", scene=scene,
    )


def cloud_trace(
    cloud: o3d.geometry.PointCloud,
    name: str = "Degraded input",
    color: str = COLORS["input"],
    max_points: int = 9000,
    scene: str = "scene",
) -> go.Scatter3d:
    points = np.asarray(cloud.points)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name=name,
        marker=dict(size=2.4, color=color, opacity=0.88), hoverinfo="skip", scene=scene,
    )


def comparison_figure(
    reference: o3d.geometry.TriangleMesh | None,
    degraded: o3d.geometry.PointCloud,
    reconstructions: dict[str, o3d.geometry.TriangleMesh],
) -> go.Figure:
    panels: list[tuple[str, str]] = []
    if reference is not None:
        panels.append(("Reference", "reference"))
    panels.append(("Degraded scan", "input"))
    panels.extend((name.upper() if name == "bpa" else name.title(), name) for name in reconstructions)
    fig = make_subplots(
        rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)],
        subplot_titles=[label for label, _ in panels], horizontal_spacing=0.015,
    )
    column = 1
    if reference is not None:
        trace = mesh_trace(reference, "Reference", COLORS["reference"])
        fig.add_trace(trace, row=1, col=column)
        column += 1
    fig.add_trace(cloud_trace(degraded), row=1, col=column)
    column += 1
    for name, mesh in reconstructions.items():
        fig.add_trace(mesh_trace(mesh, name, COLORS.get(name, "#55D6BE")), row=1, col=column)
        column += 1
    for i in range(1, len(panels) + 1):
        fig.update_layout(**{f"scene{i if i > 1 else ''}": _scene_style()})
    fig.update_layout(
        height=455, margin=dict(l=4, r=4, t=42, b=4), paper_bgcolor="#0A0F1C",
        plot_bgcolor="#0A0F1C", font=dict(color="#DCE6F5", family="Inter, sans-serif"),
        showlegend=False, uirevision="robust3d-camera",
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#A8B5CA", family="Inter, sans-serif")
    return fig


def metric_bar_figure(rows: list[dict]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Chamfer ↓", "F-score @ 1% ↑"))
    for row in rows:
        label = row["algorithm"].upper() if row["algorithm"] == "bpa" else row["algorithm"].title()
        color = COLORS.get(row["algorithm"], "#55D6BE")
        fig.add_trace(go.Bar(x=[label], y=[row.get("chamfer_x1000")], marker_color=color, showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=[label], y=[row.get("fscore_1pct")], marker_color=color, showlegend=False), row=1, col=2)
    fig.update_layout(
        height=275, margin=dict(l=35, r=15, t=38, b=28), paper_bgcolor="#111827",
        plot_bgcolor="#111827", font=dict(color="#A8B5CA", size=11), bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, linecolor="#283449")
    fig.update_yaxes(gridcolor="#202B3D", zeroline=False)
    return fig


def save_static_comparison(
    path: str | Path,
    reference: o3d.geometry.TriangleMesh,
    degraded: o3d.geometry.PointCloud,
    reconstructions: dict[str, o3d.geometry.TriangleMesh],
) -> None:
    """Save a README-friendly image from actual output geometry."""

    items: list[tuple[str, object, str]] = [("Reference", reference, COLORS["reference"]), ("Degraded input", degraded, COLORS["input"])]
    items += [(name.upper() if name == "bpa" else name.title(), mesh, COLORS[name]) for name, mesh in reconstructions.items()]
    fig = plt.figure(figsize=(3.2 * len(items), 3.4), facecolor="#080D18")
    for index, (title, geometry, color) in enumerate(items, 1):
        ax = fig.add_subplot(1, len(items), index, projection="3d", facecolor="#080D18")
        if isinstance(geometry, o3d.geometry.PointCloud):
            points = np.asarray(geometry.points)
            if len(points) > 5000:
                points = points[np.linspace(0, len(points) - 1, 5000, dtype=int)]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, c=color, alpha=0.75)
        else:
            vertices = np.asarray(geometry.vertices)
            triangles = np.asarray(geometry.triangles)
            if len(triangles) > 9000:
                triangles = triangles[np.linspace(0, len(triangles) - 1, 9000, dtype=int)]
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles,
                color=color, linewidth=0, antialiased=True, shade=True, alpha=1,
            )
        ax.set_title(title, color="#E8EEF8", fontsize=11, pad=8)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=-62)
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(pad=0.5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
