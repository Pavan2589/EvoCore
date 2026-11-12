"""Plotly charts built from engine history (no simulation logic here)."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization import theme as T

KCYC = 1000.0
CLASS_ORDER = ["Memory-bound", "Balanced", "Compute-bound"]


# ----------------------------------------------------------------------------
def history_df(engine) -> pd.DataFrame:
    if not engine.history:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in engine.history])
    df["t"] = df["cycle_end"] / KCYC
    df["t_mid"] = (df["cycle_start"] + df["cycle_end"]) / 2 / KCYC
    df["cum_instr_m"] = df["instructions"].cumsum() / 1e6
    return df


def _phase_markers(fig, engine, row=None, col=None, label=True):
    """Dotted vertical lines at actual workload phase transitions."""
    for cyc, idx, name in engine.phase_changes[1:]:
        kw = dict(row=row, col=col) if row else {}
        fig.add_vline(x=cyc / KCYC, line=dict(color=T.MUTED, width=1, dash="dot"), **kw)
        if label and not row:
            fig.add_annotation(x=cyc / KCYC, y=1.0, yref="paper", text=f"→ {name}", showarrow=False,
                               xanchor="left", yanchor="top", font=dict(size=10, color=T.MUTED))


def _reconfig_bands(fig, engine, row=None, col=None):
    for rec in engine.controller.history:
        end = min(rec.end_cycle, engine.cycle)
        kw = dict(row=row, col=col) if row else {}
        fig.add_vrect(x0=rec.start_cycle / KCYC, x1=end / KCYC, fillcolor=T.WARNING, opacity=0.18,
                      line_width=0, **kw)


def timeseries(adaptive, static, column: str, title: str, y_title: str, pct: bool = False,
               height: int = 280) -> go.Figure:
    fig = go.Figure()
    for eng, color, name in ((static, T.STATIC, "Static"), (adaptive, T.ADAPTIVE, "Adaptive")):
        if eng is None:
            continue
        df = history_df(eng)
        if df.empty:
            continue
        y = df[column] * (100 if pct else 1)
        fig.add_trace(go.Scatter(x=df["t"], y=y, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=f"%{{y:.{1 if pct else 2}f}}{'%' if pct else ''}<extra>{name}</extra>"))
    if adaptive is not None:
        _reconfig_bands(fig, adaptive)
        _phase_markers(fig, adaptive)
    T.base_layout(fig, title, height)
    fig.update_xaxes(title_text="time (k reference cycles)")
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    if pct:
        fig.update_yaxes(range=[0, 105])
    return fig


def classification_chart(adaptive, height: int = 280) -> go.Figure:
    """Validated class (line) on top of the ground-truth phase (background bands)."""
    df = history_df(adaptive)
    fig = go.Figure()
    if df.empty:
        return T.base_layout(fig, "Workload classification over time", height)
    # ground-truth phase bands
    for cyc0, cyc1, kind in _phase_spans(adaptive):
        fig.add_vrect(x0=cyc0 / KCYC, x1=cyc1 / KCYC, fillcolor=T.KIND_COLORS[kind], opacity=0.10, line_width=0)
    cls = df["classification"].replace({"Unknown": None})
    ymap = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = cls.map(ymap)
    fig.add_trace(go.Scatter(x=df["t"], y=y, mode="lines", line=dict(color=T.ADAPTIVE, width=2, shape="hv"),
                             name="Validated class", customdata=df["classification"],
                             hovertemplate="%{customdata}<extra>validated class</extra>"))
    # raw per-sample classification as markers
    hist = adaptive.classifier.history
    if hist:
        fig.add_trace(go.Scatter(
            x=[h[0] / KCYC for h in hist], y=[ymap.get(h[1]) for h in hist], mode="markers", name="Per-sample class",
            marker=dict(size=8, color=[T.CLASS_COLORS.get(h[1], T.MUTED) for h in hist],
                        line=dict(color=T.SURFACE, width=2)),
            customdata=[h[1] for h in hist], hovertemplate="%{customdata}<extra>sample</extra>"))
    _reconfig_bands(fig, adaptive)
    _phase_markers(fig, adaptive)
    T.base_layout(fig, "Workload classification over time", height)
    fig.update_yaxes(tickmode="array", tickvals=list(range(3)), ticktext=CLASS_ORDER, range=[-0.5, 2.5])
    fig.update_xaxes(title_text="time (k reference cycles) — background = true phase")
    return fig


def _phase_spans(engine):
    pcs = engine.phase_changes
    out = []
    for i, (cyc, idx, _) in enumerate(pcs):
        end = pcs[i + 1][0] if i + 1 < len(pcs) else engine.cycle
        out.append((cyc, end, engine.workload.phases[idx].kind))
    return out


def config_chart(adaptive, static=None, height: int = 420) -> go.Figure:
    df = history_df(adaptive)
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.16, horizontal_spacing=0.07,
                        subplot_titles=("CPU cores", "CPU frequency (GHz)", "LLC capacity (MB)",
                                        "Memory channels", "Prefetcher (1 = on)", "Power (W)"))
    if df.empty:
        return T.base_layout(fig, None, height)
    sdf = history_df(static) if static is not None else pd.DataFrame()
    panels = [("cores", 1, 1, 1), ("freq_ghz", 1, 2, 1), ("cache_kb", 1, 3, 1 / 1024),
              ("mem_channels", 2, 1, 1), ("prefetch", 2, 2, 1), ("power_w", 2, 3, 1)]
    for colname, r, c, scale in panels:
        if not sdf.empty:
            fig.add_trace(go.Scatter(x=sdf["t"], y=sdf[colname].astype(float) * scale, mode="lines",
                                     line=dict(color=T.STATIC, width=1.5, shape="hv"), name="Static",
                                     showlegend=(r, c) == (1, 1), hovertemplate="%{y:.2f}<extra>Static</extra>"),
                          row=r, col=c)
        fig.add_trace(go.Scatter(x=df["t"], y=df[colname].astype(float) * scale, mode="lines",
                                 line=dict(color=T.ADAPTIVE, width=2, shape="hv"), name="Adaptive",
                                 showlegend=(r, c) == (1, 1), hovertemplate="%{y:.2f}<extra>Adaptive</extra>"),
                      row=r, col=c)
        _reconfig_bands(fig, adaptive, row=r, col=c)
        _phase_markers(fig, adaptive, row=r, col=c, label=False)
    T.base_layout(fig, None, height)
    fig.update_annotations(font=dict(size=12, color=T.INK_2))
    fig.update_layout(margin=dict(l=40, r=16, t=70, b=36),
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.16, yanchor="bottom"))
    return fig


def progress_chart(adaptive, static, height: int = 300) -> go.Figure:
    fig = go.Figure()
    for eng, color, name in ((static, T.STATIC, "Static"), (adaptive, T.ADAPTIVE, "Adaptive")):
        df = history_df(eng)
        if df.empty:
            continue
        fig.add_trace(go.Scatter(x=df["t"], y=df["cum_instr_m"], name=name, mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate="%{y:.2f} M instr<extra>" + name + "</extra>"))
    total = adaptive.workload.total_instructions / 1e6
    fig.add_hline(y=total, line=dict(color=T.MUTED, dash="dot", width=1),
                  annotation_text="total work", annotation_font_color=T.MUTED)
    _reconfig_bands(fig, adaptive)
    T.base_layout(fig, "Work completed — identical workload on both architectures", height)
    fig.update_xaxes(title_text="time (k reference cycles)")
    fig.update_yaxes(title_text="instructions retired (M)", rangemode="tozero")
    return fig


def comparison_bars(cmp: dict, height: int = 360) -> go.Figure:
    s, a = cmp["static"], cmp["adaptive"]
    metrics = [
        ("Execution time (M cycles)", "total_cycles", 1e-6, "{:.3f}"),
        ("Average IPC", "avg_ipc", 1, "{:.2f}"),
        ("CPU utilisation (%)", "avg_cpu_util", 100, "{:.1f}"),
        ("Memory BW utilisation (%)", "avg_mem_bw_util", 100, "{:.1f}"),
        ("LLC hit rate (%)", "avg_hit_rate", 100, "{:.1f}"),
        ("Resource utilisation (%)", "avg_resource_util", 100, "{:.1f}"),
        ("Energy (mJ)", "energy_uj", 1e-3, "{:.2f}"),
        ("Adaptation overhead (k cycles)", "adaptation_overhead_cycles", 1e-3, "{:.1f}"),
    ]
    fig = make_subplots(rows=2, cols=4, subplot_titles=[m[0] for m in metrics], vertical_spacing=0.22,
                        horizontal_spacing=0.06)
    for i, (label, key, scale, fmt) in enumerate(metrics):
        r, c = divmod(i, 4)
        vals = [s[key] * scale, a[key] * scale]
        fig.add_trace(go.Bar(x=["Static", "Adaptive"], y=vals, marker=dict(color=[T.STATIC, T.ADAPTIVE],
                             line=dict(color=T.SURFACE, width=2), cornerradius=4),
                             text=[fmt.format(v) for v in vals], textposition="outside",
                             textfont=dict(color=T.INK_2, size=11), showlegend=False,
                             hovertemplate="%{x}: %{y:.3f}<extra>" + label + "</extra>", width=0.55),
                      row=r + 1, col=c + 1)
        top = max(vals) if max(vals) > 0 else 1
        fig.update_yaxes(range=[0, top * 1.28], showticklabels=False, row=r + 1, col=c + 1)
    T.base_layout(fig, None, height, legend=False)
    fig.update_layout(hovermode="closest", margin=dict(l=16, r=16, t=40, b=24))
    fig.update_annotations(font=dict(size=11, color=T.INK_2))
    return fig


def overhead_breakdown(summary: dict, height: int = 220) -> go.Figure:
    parts = [("Reconfiguration (lost work)", summary["reconfig_lost_cycles"], T.WARNING),
             ("Monitoring", summary["monitor_overhead_cycles"], T.ADAPTIVE),
             ("Decision-making", summary["decision_overhead_cycles"], T.NAIVE)]
    fig = go.Figure()
    for name, v, color in parts:
        fig.add_trace(go.Bar(y=["overhead"], x=[v / 1e3], name=name, orientation="h",
                             marker=dict(color=color, line=dict(color=T.SURFACE, width=2)),
                             hovertemplate=f"{name}: %{{x:,.1f}} k cycles<extra></extra>"))
    T.base_layout(fig, "Adaptation overhead breakdown (k cycles)", height)
    fig.update_layout(barmode="stack", hovermode="closest", margin=dict(l=16, r=16, t=44, b=70),
                      legend=dict(orientation="h", x=0, xanchor="left", y=-0.35, yanchor="top"))
    fig.update_yaxes(showticklabels=False)
    return fig


def sweep_chart(rows: list[dict], x: str, x_title: str, title: str, series_key: str | None = "policy",
                log_x: bool = False, height: int = 320) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    colors = {"Cost-aware adaptive": T.ADAPTIVE, "Naive adaptive": T.NAIVE}
    groups = df.groupby(series_key) if series_key else [("Cost-aware adaptive", df)]
    for name, g in groups:
        g = g.sort_values(x)
        fig.add_trace(go.Scatter(x=g[x], y=g["improvement_pct"], mode="lines+markers", name=name,
                                 line=dict(color=colors.get(name, T.ADAPTIVE), width=2),
                                 marker=dict(size=8, line=dict(color=T.SURFACE, width=2)),
                                 customdata=g["reconfigurations"],
                                 hovertemplate="%{y:+.1f}% · %{customdata} reconfigs<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=T.INK_2, width=1))
    fig.add_annotation(xref="paper", x=0.01, y=0, yanchor="top", text="below 0 = slower than static",
                       showarrow=False, font=dict(size=10, color=T.MUTED), xanchor="left")
    T.base_layout(fig, title, height)
    fig.update_xaxes(title_text=x_title, type="log" if log_x else "linear")
    if log_x:
        ticks = [v for v in (2, 3, 5, 10, 20, 30, 50, 100, 200, 500) if df[x].min() * 0.8 <= v <= df[x].max() * 1.25]
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[str(v) for v in ticks])
    fig.update_yaxes(title_text="improvement vs static (%)")
    fig.update_layout(hovermode="x unified")
    return fig
