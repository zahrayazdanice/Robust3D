"""Interactive scientific dashboard backed by the real reconstruction pipeline."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update

from .config import DegradationConfig, PipelineConfig, PreprocessConfig, ReconstructionConfig
from .datasets import DATASETS
from .io import decode_dash_upload, mesh_to_ascii_ply
from .pipeline import prepare_case, run_prepared
from .visualization import comparison_figure, metric_bar_figure

_CACHE: OrderedDict[str, dict] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 8


def _cache_result(value: dict) -> str:
    key = uuid.uuid4().hex
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return key


def _empty_figure(message: str, height: int = 455) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color="#94A3B8", size=14))
    fig.update_layout(height=height, paper_bgcolor="#0A0F1C", plot_bgcolor="#0A0F1C", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _control_label(title: str, value_id: str, value: str) -> html.Div:
    return html.Div([html.Label(title), html.Span(value, id=value_id, className="control-value")], className="control-label")


def _slider(control_id: str, minimum: float, maximum: float, step: float, value: float, marks: dict | None = None) -> dcc.Slider:
    return dcc.Slider(id=control_id, min=minimum, max=maximum, step=step, value=value, marks=marks or {}, tooltip={"placement": "bottom", "always_visible": False})


def _kpi(label: str, value: str, note: str, accent: str = "mint") -> html.Div:
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(note, className="kpi-note"),
    ], className=f"kpi-card accent-{accent}")


def _metric_table_columns() -> list[dict]:
    return [
        {"name": "Method", "id": "method"},
        {"name": "Chamfer ×1000 ↓", "id": "chamfer", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "F-score ↑", "id": "fscore", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "Normals ↑", "id": "normals", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "H95 ×1000 ↓", "id": "h95", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Triangles", "id": "triangles", "type": "numeric", "format": {"specifier": ",d"}},
        {"name": "Reconstruct", "id": "runtime"},
    ]


def _benchmark_figure() -> go.Figure:
    metrics_path = Path("results/metrics.csv")
    if not metrics_path.exists():
        return _empty_figure("Run `robust3d experiment` to populate the benchmark explorer.", 300)
    frame = pd.read_csv(metrics_path)
    frame = frame[frame["status"] == "ok"]
    if frame.empty:
        return _empty_figure("No successful benchmark rows are available yet.", 300)
    grouped = frame.groupby(["scenario", "preprocessing"], sort=False)["fscore_1pct"].mean().reset_index()
    fig = go.Figure()
    for mode, color in (("baseline", "#6C8BFF"), ("adaptive", "#55D6BE")):
        subset = grouped[grouped["preprocessing"] == mode]
        fig.add_bar(name=mode.title(), x=subset["scenario"], y=subset["fscore_1pct"], marker_color=color)
    fig.update_layout(
        barmode="group", height=300, margin=dict(l=42, r=18, t=20, b=42),
        paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#A8B5CA"),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title="Mean F-score @ 1%", gridcolor="#202B3D", range=[0, 1]),
        xaxis=dict(title="Measured degradation scenario", showgrid=False),
    )
    return fig


def _layout() -> html.Div:
    dataset_options = [{"label": value.label, "value": key} for key, value in DATASETS.items()]
    return html.Div([
        dcc.Store(id="result-key"),
        dcc.Store(id="metric-figure-store", data=_empty_figure("Run a reconstruction to see metrics.", 275).to_dict()),
        dcc.Store(id="metric-data-store", data=[]),
        dcc.Download(id="mesh-download"),
        html.Header([
            html.Div([
                html.Div("R³", className="brand-mark"),
                html.Div("ROBUST3D", className="brand-name"),
            ], className="brand"),
            html.Div([html.Span(className="live-dot"), "LOCAL COMPUTE"], className="privacy-chip"),
            html.A("View methodology", href="#methodology", className="text-link"),
        ], className="topbar"),
        html.Main([
            html.Section([
                html.Div("SURFACE RECONSTRUCTION / 01", className="eyebrow"),
                html.H1(["Interactive 3D ", html.Span("reconstruction demo.")]),
                html.P("Compare three classical reconstruction families under controlled sparsity, noise, occlusion, and outliers. Every mesh and metric below is computed when you press Run."),
                html.Div([
                    html.Span("POISSON"), html.Span("BALL PIVOTING"), html.Span("ALPHA SHAPES"), html.Span("ADAPTIVE MLS + MAD"),
                ], className="method-strip"),
            ], className="hero"),

            html.Div([
                html.Aside([
                    html.Div([html.Span("01"), html.H2("Input geometry")], className="section-heading"),
                    html.Label("Reference object", className="field-label"),
                    dcc.Dropdown(id="dataset", options=dataset_options, value="bunny", clearable=False, searchable=False, className="dark-dropdown"),
                    dcc.Upload(
                        id="upload", multiple=False,
                        children=html.Div([html.Div("＋", className="upload-icon"), html.Div([html.Strong("Drop your geometry"), html.Span("PLY · OBJ · STL · OFF · XYZ · PCD")])]),
                        className="upload-box", accept=".ply,.obj,.stl,.off,.xyz,.pts,.pcd",
                    ),
                    html.Div(id="upload-name", className="upload-name"),
                    _control_label("Surface samples", "points-value", "3,500"),
                    _slider("points", 1000, 8000, 500, 3500, {1000: "1k", 3500: "3.5k", 8000: "8k"}),

                    html.Div([html.Span("02"), html.H2("Degrade the scan")], className="section-heading spaced"),
                    _control_label("Sampling density", "density-value", "65%"),
                    _slider("density", 20, 100, 5, 65, {20: "20", 60: "60", 100: "100%"}),
                    _control_label("Gaussian noise σ", "noise-value", "0.60% diag"),
                    _slider("noise", 0, 2.0, 0.1, 0.6, {0: "0", 1: "1%", 2: "2%"}),
                    _control_label("Missing cap", "missing-value", "12%"),
                    _slider("missing", 0, 40, 2, 12, {0: "0", 20: "20", 40: "40%"}),
                    _control_label("Uniform outliers", "outliers-value", "4%"),
                    _slider("outliers", 0, 20, 1, 4, {0: "0", 10: "10", 20: "20%"}),

                    html.Div([html.Span("03"), html.H2("Reconstruct")], className="section-heading spaced"),
                    html.Label("Preprocessing", className="field-label"),
                    dcc.RadioItems(
                        id="preprocessing",
                        options=[{"label": "Baseline", "value": "baseline"}, {"label": "Adaptive", "value": "adaptive"}],
                        value="adaptive", inline=True, className="segmented",
                    ),
                    html.Label("Algorithms", className="field-label top-gap"),
                    dcc.Checklist(
                        id="algorithms",
                        options=[
                            {"label": "Poisson", "value": "poisson"},
                            {"label": "Ball Pivoting", "value": "bpa"},
                            {"label": "Alpha Shapes", "value": "alpha"},
                        ],
                        value=["poisson", "bpa", "alpha"], className="algorithm-checks",
                    ),
                    html.Details([
                        html.Summary("Algorithm parameters"),
                        _control_label("Poisson depth", "depth-value", "7"),
                        _slider("depth", 5, 9, 1, 7, {5: "5", 7: "7", 9: "9"}),
                        _control_label("Alpha / spacing", "alpha-value", "4.5×"),
                        _slider("alpha", 2.0, 8.0, 0.5, 4.5, {2: "2×", 5: "5×", 8: "8×"}),
                        html.Label("BPA radii / spacing", className="field-label top-gap"),
                        dcc.Input(id="bpa-radii", value="1.5, 2.5, 4.0", type="text", debounce=True, className="text-input"),
                        html.Label("Random seed", className="field-label top-gap"),
                        dcc.Input(id="seed", value=7, type="number", min=0, step=1, className="text-input"),
                    ], className="advanced"),
                    html.Button([html.Span("Run reconstruction", className="run-label"), html.Span("→", className="run-arrow")], id="run-button", className="run-button", n_clicks=0),
                ], className="control-panel"),

                html.Section([
                    html.Div([
                        html.Div([html.Span(className="status-light"), html.Span("READY", id="run-status")], className="run-status"),
                        html.Div([html.Span("Drag to orbit"), html.Span("Scroll to zoom"), html.Span("Double-click to reset")], className="viewer-hints"),
                    ], className="viewer-toolbar"),
                    html.Div(id="error-banner", className="error-banner hidden"),
                    dcc.Loading(
                        id="scene-loading", type="circle", color="#55D6BE",
                        children=dcc.Graph(id="scene", figure=_empty_figure("Preparing default experiment…"), config={"displaylogo": False, "scrollZoom": True, "responsive": True}, className="scene-graph"),
                    ),
                    html.Div(id="kpis", className="kpi-grid", children=[
                        _kpi("Input points", "—", "after degradation", "orange"),
                        _kpi("Best Chamfer", "—", "lower is better", "mint"),
                        _kpi("Best F-score", "—", "higher is better", "violet"),
                        _kpi("Wall time", "—", "shared preprocessing", "blue"),
                    ]),
                    html.Div([
                        html.Div([html.H3("Export computed mesh"), html.P("PLY · normalized object coordinates")]),
                        dcc.Dropdown(id="download-algo", options=[], value=None, clearable=False, searchable=False, className="compact-dropdown"),
                        html.Button("Download PLY", id="download-button", className="secondary-button"),
                    ], className="export-row"),
                ], className="viewer-panel"),
            ], className="lab-grid"),

            html.Section([
                dcc.Tabs(id="analysis-tabs", value="metrics", className="tabs", children=[
                    dcc.Tab(label="LIVE METRICS", value="metrics", className="tab", selected_className="tab selected"),
                    dcc.Tab(label="BENCHMARK", value="benchmark", className="tab", selected_className="tab selected"),
                    dcc.Tab(label="METHOD", value="method", className="tab", selected_className="tab selected"),
                ]),
                html.Div(id="tab-content"),
            ], className="analysis-section", id="methodology"),
            html.Footer([
                html.Div(html.Strong("ROBUST3D")),
                html.Div("Open3D · NumPy · SciPy · Plotly Dash"),
            ]),
        ]),
    ], className="app-shell")


def create_app() -> Dash:
    assets_folder = Path(__file__).with_name("assets")
    app = Dash(
        __name__,
        title="Robust3D · Geometry Lab",
        update_title="Reconstructing…",
        suppress_callback_exceptions=True,
        assets_folder=str(assets_folder),
    )
    app.layout = _layout()

    @callback(
        Output("points-value", "children"), Output("density-value", "children"),
        Output("noise-value", "children"), Output("missing-value", "children"),
        Output("outliers-value", "children"), Output("depth-value", "children"),
        Output("alpha-value", "children"),
        Input("points", "value"), Input("density", "value"), Input("noise", "value"),
        Input("missing", "value"), Input("outliers", "value"), Input("depth", "value"), Input("alpha", "value"),
    )
    def update_labels(points, density, noise, missing, outliers, depth, alpha):
        return f"{int(points):,}", f"{density:.0f}%", f"{noise:.2f}% diag", f"{missing:.0f}%", f"{outliers:.0f}%", str(depth), f"{alpha:.1f}×"

    @callback(Output("upload-name", "children"), Input("upload", "filename"))
    def show_upload_name(filename):
        return f"Using upload: {filename}" if filename else "Or use a built-in reference above"

    @callback(
        Output("tab-content", "children"), Input("analysis-tabs", "value"),
        Input("metric-figure-store", "data"), Input("metric-data-store", "data"),
    )
    def render_tab(tab, metric_figure, metric_rows):
        if tab == "metrics":
            return html.Div([
                html.Div([
                    html.Div([html.Div("CURRENT RUN", className="eyebrow"), html.H2("Geometry quality"), html.P("Distances are normalized by the clean reference diagonal.")], className="analysis-copy"),
                    dcc.Graph(figure=metric_figure or _empty_figure("Run a reconstruction to see metrics.", 275), config={"displayModeBar": False}),
                ], className="analysis-top"),
                html.Div(
                    dash_table.DataTable(
                        columns=_metric_table_columns(), data=metric_rows or [],
                        sort_action="native", style_as_list_view=True,
                        style_table={"overflowX": "auto"},
                        style_header={"backgroundColor": "#0E1626", "color": "#7F8FA8", "border": "none", "fontWeight": "600"},
                        style_cell={"backgroundColor": "#111827", "color": "#DDE7F5", "border": "none", "borderBottom": "1px solid #202B3D", "padding": "14px 12px", "fontFamily": "Inter, sans-serif", "fontSize": 12},
                    ),
                    className="metric-table",
                ),
            ], className="tab-panel")
        if tab == "benchmark":
            return html.Div([
                html.Div([html.Div("REPRODUCIBLE BATCH", className="eyebrow"), html.H2("Robustness across degradation regimes"), html.P("This chart reads results/metrics.csv produced by the experiment runner.")], className="analysis-copy"),
                dcc.Graph(figure=_benchmark_figure(), config={"displayModeBar": False}),
            ], className="tab-panel benchmark-panel")
        return html.Div([
            html.Div([html.Div("GEOMETRIC IMPROVEMENT", className="eyebrow"), html.H2("Density-aware preprocessing"), html.P("A single fixed radius is brittle when samples thin out or outliers expand local neighborhoods. Robust3D estimates median k-NN spacing, removes high-scale points with a median-absolute-deviation rule, equalizes dense voxels, performs one conservative moving-least-squares projection, then estimates consistently oriented normals at a spacing-scaled radius.")], className="method-copy"),
            html.Div([
                html.Div([html.Span("01"), html.H3("Robust scale"), html.P("Median local k-NN distance resists a minority of far outliers.")]),
                html.Div([html.Span("02"), html.H3("MAD rejection"), html.P("A distribution-aware threshold replaces an object-unit cutoff.")]),
                html.Div([html.Span("03"), html.H3("Local tangent fit"), html.P("Weighted PCA removes normal-direction noise while limiting gap shrinkage.")]),
                html.Div([html.Span("04"), html.H3("Fair evaluation"), html.P("The same degraded sample and seed are shared across algorithms.")]),
            ], className="method-grid"),
        ], className="tab-panel")

    @callback(
        Output("scene", "figure"), Output("metric-figure-store", "data"), Output("metric-data-store", "data"),
        Output("run-status", "children"), Output("kpis", "children"), Output("result-key", "data"),
        Output("download-algo", "options"), Output("download-algo", "value"),
        Output("error-banner", "children"), Output("error-banner", "className"),
        Input("run-button", "n_clicks"),
        State("dataset", "value"), State("upload", "contents"), State("upload", "filename"),
        State("points", "value"), State("density", "value"), State("noise", "value"),
        State("missing", "value"), State("outliers", "value"), State("preprocessing", "value"),
        State("algorithms", "value"), State("depth", "value"), State("alpha", "value"),
        State("bpa-radii", "value"), State("seed", "value"),
        running=[(Output("run-button", "disabled"), True, False), (Output("run-button", "className"), "run-button running", "run-button")],
    )
    def run_reconstruction(_, dataset, upload_contents, filename, points, density, noise, missing, outliers, preprocessing, algorithms, depth, alpha, bpa_radii, seed):
        try:
            if not algorithms:
                raise ValueError("Select at least one reconstruction algorithm.")
            factors = tuple(float(value.strip()) for value in str(bpa_radii).split(",") if value.strip())
            if not factors or min(factors) <= 0:
                raise ValueError("BPA radii must be a comma-separated list of positive numbers.")
            reference = None
            source_cloud = None
            object_name = dataset
            if upload_contents and filename:
                reference, uploaded_cloud = decode_dash_upload(upload_contents, filename)
                object_name = f"upload:{filename}"
                if reference is None:
                    source_cloud = uploaded_cloud

            base = PipelineConfig(
                object_name=object_name,
                n_points=int(points),
                evaluation_samples=3500,
                degradation=DegradationConfig(
                    density=float(density) / 100, noise_sigma=float(noise) / 100,
                    missing_fraction=float(missing) / 100, outlier_fraction=float(outliers) / 100,
                    seed=int(seed or 0),
                ),
                preprocessing=PreprocessConfig(mode=preprocessing),
                reconstruction=ReconstructionConfig(poisson_depth=int(depth)),
            )
            prepared = prepare_case(base, reference_mesh=reference, source_cloud=source_cloud)
            meshes = {}
            metric_rows = []
            graph_rows = []
            errors = []
            results = {}
            for algorithm in algorithms:
                recon_config = ReconstructionConfig(
                    algorithm=algorithm, poisson_depth=int(depth), alpha_factor=float(alpha),
                    bpa_radius_factors=factors,
                )
                run_config = PipelineConfig(
                    object_name=base.object_name, n_points=base.n_points,
                    degradation=base.degradation, preprocessing=base.preprocessing,
                    reconstruction=recon_config, evaluation_samples=base.evaluation_samples,
                )
                try:
                    result = run_prepared(prepared, run_config, recon_config)
                    meshes[algorithm] = result.reconstructed_mesh
                    results[algorithm] = result
                    metrics = result.metrics
                    metric_rows.append({
                        "method": "BPA" if algorithm == "bpa" else algorithm.title(),
                        "chamfer": metrics.chamfer_x1000 if metrics else None,
                        "fscore": metrics.fscore_1pct if metrics else None,
                        "normals": metrics.normal_consistency if metrics else None,
                        "h95": metrics.hausdorff95_x1000 if metrics else None,
                        "triangles": len(result.reconstructed_mesh.triangles),
                        "runtime": f"{result.reconstruction_report.seconds:.2f} s",
                    })
                    graph_rows.append({
                        "algorithm": algorithm,
                        "chamfer_x1000": metrics.chamfer_x1000 if metrics else None,
                        "fscore_1pct": metrics.fscore_1pct if metrics else None,
                    })
                except Exception as exc:
                    errors.append(f"{algorithm.upper()}: {exc}")
            if not meshes:
                raise RuntimeError("All selected algorithms failed. " + " · ".join(errors))
            figure = comparison_figure(prepared.reference_mesh, prepared.degraded_cloud, meshes)
            metric_figure = metric_bar_figure(graph_rows) if prepared.reference_mesh is not None else _empty_figure("No reference mesh: visual and topology inspection only.", 275)
            valid_metrics = [result.metrics for result in results.values() if result.metrics]
            best_chamfer = min((m.chamfer_x1000 for m in valid_metrics), default=None)
            best_fscore = max((m.fscore_1pct for m in valid_metrics), default=None)
            wall = prepared.preparation_seconds + sum(r.reconstruction_report.seconds for r in results.values())
            kpis = [
                _kpi("Input points", f"{prepared.degradation_report.final_points:,}", f"{prepared.preprocessing_report.rejected_outliers:,} rejected", "orange"),
                _kpi("Best Chamfer", f"{best_chamfer:.3f}" if best_chamfer is not None else "N/A", "×1000 normalized", "mint"),
                _kpi("Best F-score", f"{best_fscore:.3f}" if best_fscore is not None else "N/A", "at 1% diagonal", "violet"),
                _kpi("Wall time", f"{wall:.2f}s", f"{len(meshes)} method{'s' if len(meshes) != 1 else ''}", "blue"),
            ]
            key = _cache_result({"meshes": meshes, "object": object_name})
            options = [{"label": "BPA" if name == "bpa" else name.title(), "value": name} for name in meshes]
            error_text = "Some methods failed · " + " · ".join(errors) if errors else ""
            return figure, metric_figure.to_dict(), metric_rows, "COMPLETE", kpis, key, options, next(iter(meshes)), error_text, "error-banner" if errors else "error-banner hidden"
        except Exception as exc:
            return no_update, no_update, no_update, "ERROR", no_update, no_update, no_update, no_update, str(exc), "error-banner"

    @callback(
        Output("mesh-download", "data"), Input("download-button", "n_clicks"),
        State("result-key", "data"), State("download-algo", "value"), prevent_initial_call=True,
    )
    def download_mesh(_, key, algorithm):
        if not key or not algorithm:
            return no_update
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if not cached or algorithm not in cached["meshes"]:
            return no_update
        payload = mesh_to_ascii_ply(cached["meshes"][algorithm])

        def writer(stream):
            stream.write(payload)

        return dcc.send_bytes(writer, f"robust3d_{algorithm}.ply")

    return app
