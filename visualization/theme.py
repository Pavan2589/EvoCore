"""Dark technical theme shared by all dashboard figures (validated dark palette)."""

from __future__ import annotations

import plotly.graph_objects as go

# Surfaces & ink
PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
SURFACE_2 = "#232321"
GRID = "#2c2c2a"
AXIS = "#383835"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
BORDER = "rgba(255,255,255,0.10)"

# Categorical series (fixed order, dark steps)
ADAPTIVE = "#3987e5"     # slot 1 blue
STATIC = "#d95926"       # slot 2 orange
NAIVE = "#9085e9"        # slot 7 violet

# Workload kinds / classes (one entity -> one colour everywhere)
KIND_COLORS = {"compute": "#199e70", "memory": "#d55181", "mixed": "#c98500"}
CLASS_COLORS = {"Compute-bound": "#199e70", "Memory-bound": "#d55181",
                "Balanced": "#c98500", "Unknown": MUTED, "-": MUTED}
KIND_TO_CLASS = {"compute": "Compute-bound", "memory": "Memory-bound", "mixed": "Balanced"}

# Status (reserved; always paired with a text label)
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

ACTIVITY_COLORS = {
    "normal": GOOD,
    "monitoring": ADAPTIVE,
    "deciding": NAIVE,
    "reconfiguring": WARNING,
    "overhead": CRITICAL,
}
ACTIVITY_LABELS = {
    "normal": "Normal execution",
    "monitoring": "Monitoring sample",
    "deciding": "Decision-making",
    "reconfiguring": "Reconfiguring",
    "overhead": "Adaptation overhead",
}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def base_layout(fig: go.Figure, title: str | None = None, height: int = 300,
                legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=12),
        title=dict(text=title, font=dict(size=14, color=INK), x=0.01, xanchor="left") if title else None,
        height=height,
        margin=dict(l=56, r=16, t=44 if title else 16, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=AXIS, font=dict(color=INK, family=FONT)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    return fig
