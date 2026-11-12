"""
Live architecture block diagram (Plotly shapes).

Top row  : datapath   Workload -> CPU -> Cache -> Memory Ctrl -> Main Memory
Bottom row: control   Monitor -> Characterization -> Decision -> Reconfig Ctrl
Dashed blue arrows    : performance counters flowing to the monitor
Solid arrows          : data flow (top) / control flow (bottom)
Orange arrows         : reconfiguration commands back to the datapath

Block borders encode the current activity (with a text legend):
normal (green), monitoring (blue), decision (violet), reconfiguring (amber),
adaptation overhead (red).
"""

from __future__ import annotations

import plotly.graph_objects as go

from simulation.models import fmt_cache
from visualization import theme as T

W, H = 100.0, 58.0
TOP_Y0, TOP_Y1 = 30.0, 52.0
BOT_Y0, BOT_Y1 = 3.0, 19.0

TOP_BLOCKS = {          # name: (x0, x1)
    "workload": (1.0, 17.0),
    "cpu": (21.0, 41.0),
    "cache": (45.0, 62.0),
    "memctrl": (66.0, 82.0),
    "dram": (86.0, 99.0),
}
BOT_BLOCKS = {
    "monitor": (6.0, 26.0),
    "classifier": (30.0, 50.0),
    "decision": (54.0, 74.0),
    "reconfig": (78.0, 98.0),
}


def _rect(fig, x0, y0, x1, y1, border, fill=T.SURFACE_2, width=2, dash=None):
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=border, width=width, dash=dash),
                  fillcolor=fill, layer="below")


def _text(fig, x, y, text, size=11, color=T.INK_2, anchor="center", bold=False):
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False,
                       font=dict(size=size, color=color, family=T.FONT), xanchor=anchor, yanchor="middle")


def _arrow(fig, x0, y0, x1, y1, color, dash=False, width=1.6):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=width, arrowcolor=color,
                       text="", standoff=1, startstandoff=1)
    if dash:  # plotly annotations cannot dash; overlay a dashed line segment
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=color, width=width, dash="dot"))


def _status_color(active: bool, kind: str) -> str:
    return T.ACTIVITY_COLORS[kind] if active else T.AXIS


def build_architecture_figure(engine, height: int = 430) -> go.Figure:
    fig = go.Figure()
    last = engine.history[-1] if engine.history else None
    activity = last.activity if last else "normal"
    rec = engine.controller.active
    cfg = engine.config
    reconf_fields = set(rec.changed) if rec else set()
    overhead = bool(last and (last.monitor_overhead_cycles + last.decision_overhead_cycles
                              + last.reconfig_lost_cycles) > 0)

    def dp_border(fields: set) -> str:
        if rec and reconf_fields & fields:
            return T.WARNING
        return T.GOOD if not engine.finished else T.AXIS

    # ---------------- datapath ----------------
    wl = engine.workload
    phase = wl.current_phase
    kind_color = T.KIND_COLORS.get(phase.kind, T.MUTED)
    x0, x1 = TOP_BLOCKS["workload"]
    _rect(fig, x0, TOP_Y0, x1, TOP_Y1, T.GOOD if not engine.finished else T.AXIS)
    cx = (x0 + x1) / 2
    _text(fig, cx, 49.5, "Workload Generator", 11, T.INK, bold=True)
    _text(fig, cx, 45.5, f"Phase {wl.phase_index + 1}/{len(wl.phases)}", 10, T.MUTED)
    _text(fig, cx, 41.5, phase.label, 12, kind_color, bold=True)
    prog = wl.retired / wl.total_instructions
    fig.add_shape(type="rect", x0=x0 + 1.5, y0=35, x1=x1 - 1.5, y1=37, line=dict(width=0), fillcolor=T.GRID)
    fig.add_shape(type="rect", x0=x0 + 1.5, y0=35, x1=x0 + 1.5 + (x1 - x0 - 3) * prog, y1=37,
                  line=dict(width=0), fillcolor=T.ADAPTIVE)
    _text(fig, cx, 32.5, f"{wl.retired / 1e6:,.1f} / {wl.total_instructions / 1e6:,.1f} M instr", 9, T.MUTED)

    # CPU
    x0, x1 = TOP_BLOCKS["cpu"]
    _rect(fig, x0, TOP_Y0, x1, TOP_Y1, dp_border({"cores", "freq_ghz"}))
    cx = (x0 + x1) / 2
    _text(fig, cx, 49.5, "CPU Execution Units", 11, T.INK, bold=True)
    cw, gap = 3.6, 0.9
    start = cx - (4 * cw + 3 * gap) / 2
    util = last.cpu_util if last else 0.0
    for i in range(8):
        r, c = divmod(i, 4)
        xx = start + c * (cw + gap)
        yy = 43.0 - r * (cw + gap * 0.8)
        on = i < cfg.cores
        pending = rec is not None and "cores" in reconf_fields and (
            (i < rec.new.cores) != (i < rec.old.cores))
        fill = T.ADAPTIVE if on else T.SURFACE
        line = T.WARNING if pending else (T.ADAPTIVE if on else T.AXIS)
        fig.add_shape(type="rect", x0=xx, y0=yy, x1=xx + cw, y1=yy + cw * 0.75,
                      line=dict(color=line, width=1.5), fillcolor=fill, opacity=1.0 if on else 0.6)
    _text(fig, cx, 34.6, f"{cfg.cores} cores @ {cfg.freq_ghz:.1f} GHz", 11, T.INK)
    _text(fig, cx, 32.0, f"util {util:.0%} · IPC {last.ipc if last else 0:.2f}", 9, T.MUTED)

    # Cache
    x0, x1 = TOP_BLOCKS["cache"]
    _rect(fig, x0, TOP_Y0, x1, TOP_Y1, dp_border({"cache_kb", "policy"}))
    cx = (x0 + x1) / 2
    _text(fig, cx, 49.5, "Cache Hierarchy", 11, T.INK, bold=True)
    _text(fig, cx, 46.2, "L1 32 KB/core (private)", 9, T.MUTED)
    bx0, bx1 = x0 + 1.5, x1 - 1.5
    max_kb = 4096.0
    cap_frac = cfg.cache_kb / max_kb
    res_frac = min(engine.cache.resident_kb, cfg.cache_kb) / max_kb
    fig.add_shape(type="rect", x0=bx0, y0=39.5, x1=bx1, y1=43.5, line=dict(color=T.AXIS, width=1),
                  fillcolor=T.SURFACE)
    fig.add_shape(type="rect", x0=bx0, y0=39.5, x1=bx0 + (bx1 - bx0) * cap_frac, y1=43.5,
                  line=dict(color=T.KIND_COLORS["mixed"], width=1.5), fillcolor="rgba(201,133,0,0.18)")
    fig.add_shape(type="rect", x0=bx0, y0=39.5, x1=bx0 + (bx1 - bx0) * res_frac, y1=43.5,
                  line=dict(width=0), fillcolor="rgba(201,133,0,0.55)")
    _text(fig, cx, 37.0, f"LLC {fmt_cache(cfg.cache_kb)} · {cfg.policy}", 11, T.INK)
    _text(fig, cx, 34.4, f"hit {last.hit_rate if last else 0:.0%} · warm {engine.cache.resident_kb / 1024:.1f} MB", 9, T.MUTED)
    _text(fig, cx, 32.0, "shared, way-reconfigurable", 9, T.MUTED)

    # Memory controller
    x0, x1 = TOP_BLOCKS["memctrl"]
    _rect(fig, x0, TOP_Y0, x1, TOP_Y1, dp_border({"mem_channels", "prefetch"}))
    cx = (x0 + x1) / 2
    _text(fig, cx, 49.5, "Memory Controller", 11, T.INK, bold=True)
    bw = 2.2
    for i in range(4):
        xx = cx - 2 * (bw + 0.8) + i * (bw + 0.8) + 0.4
        on = i < cfg.mem_channels
        pending = rec is not None and "mem_channels" in reconf_fields and (
            (i < rec.new.mem_channels) != (i < rec.old.mem_channels))
        fig.add_shape(type="rect", x0=xx, y0=39.0, x1=xx + bw, y1=46.0,
                      line=dict(color=T.WARNING if pending else (T.KIND_COLORS["memory"] if on else T.AXIS), width=1.5),
                      fillcolor=T.KIND_COLORS["memory"] if on else T.SURFACE, opacity=1.0 if on else 0.6)
    _text(fig, cx, 37.0, f"{cfg.mem_channels} channel{'s' if cfg.mem_channels > 1 else ''} · PF {'ON' if cfg.prefetch else 'off'}",
          11, T.INK)
    _text(fig, cx, 34.4, f"BW util {last.mem_bw_util if last else 0:.0%}", 9, T.MUTED)
    _text(fig, cx, 32.0, f"queue {last.queue_util if last else 0:.0%}", 9, T.MUTED)

    # DRAM
    x0, x1 = TOP_BLOCKS["dram"]
    _rect(fig, x0, TOP_Y0, x1, TOP_Y1, T.GOOD if not engine.finished else T.AXIS)
    cx = (x0 + x1) / 2
    _text(fig, cx, 49.5, "Main Memory", 11, T.INK, bold=True)
    _text(fig, cx, 44.0, f"{last.mem_latency_ns if last else 75:.0f} ns", 14, T.INK, bold=True)
    _text(fig, cx, 40.5, "access latency", 9, T.MUTED)
    _text(fig, cx, 36.0, f"{last.mem_bw_gbps if last else 0:.1f} GB/s", 11, T.INK)
    _text(fig, cx, 33.0, f"of {cfg.mem_channels * 12.8:.1f} peak", 9, T.MUTED)

    # data-flow arrows
    order = ["workload", "cpu", "cache", "memctrl", "dram"]
    for a, b in zip(order, order[1:]):
        _arrow(fig, TOP_BLOCKS[a][1], 41, TOP_BLOCKS[b][0], 41, T.INK_2, width=2)

    # ---------------- control path ----------------
    adaptive = engine.adaptive
    mon_active = activity == "monitoring" or activity == "deciding"
    x0, x1 = BOT_BLOCKS["monitor"]
    _rect(fig, x0, BOT_Y0, x1, BOT_Y1, _status_color(adaptive and mon_active, "monitoring"),
          dash=None if adaptive else "dot")
    cx = (x0 + x1) / 2
    _text(fig, cx, 16.5, "Runtime Monitoring Unit", 11, T.INK, bold=True)
    smp = engine.monitor.samples[-1] if engine.monitor.samples else None
    if adaptive:
        _text(fig, cx, 12.3, f"{len(engine.monitor.samples)} samples · every {engine.params.sampling_interval} steps", 9, T.MUTED)
        if smp:
            _text(fig, cx, 8.8, f"CPU {smp.cpu_util:.0%} · BW {smp.mem_bw_util:.0%}", 10, T.INK_2)
            _text(fig, cx, 5.6, f"miss {smp.miss_rate:.0%} · stall {smp.mem_stall_frac:.0%}", 10, T.INK_2)
    else:
        _text(fig, cx, 9.5, "not present in static design", 10, T.MUTED)

    x0, x1 = BOT_BLOCKS["classifier"]
    stable = engine.classifier.stable if adaptive else "-"
    _rect(fig, x0, BOT_Y0, x1, BOT_Y1, _status_color(adaptive and mon_active, "monitoring"),
          dash=None if adaptive else "dot")
    cx = (x0 + x1) / 2
    _text(fig, cx, 16.5, "Workload Characterization", 11, T.INK, bold=True)
    if adaptive:
        _text(fig, cx, 11.0, stable, 14, T.CLASS_COLORS.get(stable, T.MUTED), bold=True)
        hist = engine.classifier.history
        raw = hist[-1][1] if hist else "-"
        _text(fig, cx, 6.2, f"last sample: {raw} · confirm ×{engine.params.confirm_samples}", 9, T.MUTED)
    else:
        _text(fig, cx, 9.5, "-", 10, T.MUTED)

    x0, x1 = BOT_BLOCKS["decision"]
    _rect(fig, x0, BOT_Y0, x1, BOT_Y1, _status_color(adaptive and activity == "deciding", "deciding"),
          dash=None if adaptive else "dot")
    cx = (x0 + x1) / 2
    _text(fig, cx, 16.5, "Autonomous Decision Engine", 11, T.INK, bold=True)
    d = engine.decider.decisions[-1] if adaptive and engine.decider.decisions else None
    if d:
        col = T.WARNING if d.action == "reconfigure" else T.INK_2
        _text(fig, cx, 11.5, d.action.upper(), 13, col, bold=True)
        _text(fig, cx, 7.6, f"pred. gain {d.predicted_gain:+.1%} · {d.candidates_evaluated} candidates", 9, T.MUTED)
        _text(fig, cx, 5.0, f"horizon ≈ {d.horizon_steps:.0f} steps", 9, T.MUTED)
    else:
        _text(fig, cx, 9.5, "idle" if adaptive else "-", 10, T.MUTED)

    x0, x1 = BOT_BLOCKS["reconfig"]
    _rect(fig, x0, BOT_Y0, x1, BOT_Y1, _status_color(rec is not None, "reconfiguring"),
          dash=None if adaptive else "dot")
    cx = (x0 + x1) / 2
    _text(fig, cx, 16.5, "Reconfiguration Controller", 11, T.INK, bold=True)
    if rec:
        frac = min(1.0, max(0.0, (engine.cycle - rec.start_cycle) / rec.duration))
        _text(fig, cx, 12.4, f"#{rec.id} in progress · {', '.join(_short(f) for f in rec.changed)}", 9, T.WARNING)
        fig.add_shape(type="rect", x0=x0 + 2, y0=8.2, x1=x1 - 2, y1=9.8, line=dict(width=0), fillcolor=T.GRID)
        fig.add_shape(type="rect", x0=x0 + 2, y0=8.2, x1=x0 + 2 + (x1 - x0 - 4) * frac, y1=9.8,
                      line=dict(width=0), fillcolor=T.WARNING)
        _text(fig, cx, 5.4, f"throughput ×{rec.throughput_factor:.2f} for {rec.duration:,.0f} cyc", 9, T.MUTED)
    else:
        n = len(engine.controller.history)
        _text(fig, cx, 11.0, "idle" if adaptive else "-", 12, T.INK_2)
        if adaptive:
            _text(fig, cx, 6.2, f"{n} reconfiguration{'s' if n != 1 else ''} so far", 9, T.MUTED)

    # control-flow arrows (bottom row)
    border = [("monitor", "classifier"), ("classifier", "decision"), ("decision", "reconfig")]
    for a, b in border:
        _arrow(fig, BOT_BLOCKS[a][1], 11, BOT_BLOCKS[b][0], 11, T.INK_2 if adaptive else T.AXIS, width=2)

    # Two orthogonal buses between the rows:
    #   counter bus (blue, dashed, y=26.5): datapath blocks -> Runtime Monitoring Unit
    #   command bus (amber, y=22.5):        Reconfiguration Controller -> datapath blocks
    mcol = T.ADAPTIVE if adaptive else T.AXIS
    ccol = T.WARNING if rec else (T.SERIOUS if adaptive else T.AXIS)
    taps = [(TOP_BLOCKS[n][0] + TOP_BLOCKS[n][1]) / 2 for n in ("cpu", "cache", "memctrl")]
    mon_x = (BOT_BLOCKS["monitor"][0] + BOT_BLOCKS["monitor"][1]) / 2 - 3
    rc_x = (BOT_BLOCKS["reconfig"][0] + BOT_BLOCKS["reconfig"][1]) / 2 + 3
    y_cnt, y_cmd = 26.5, 22.5
    dot = dict(color=mcol, width=1.4, dash="dot")
    for x in taps:
        fig.add_shape(type="line", x0=x - 3, y0=TOP_Y0, x1=x - 3, y1=y_cnt, line=dot)
    fig.add_shape(type="line", x0=mon_x, y0=y_cnt, x1=taps[-1] - 3, y1=y_cnt, line=dot)
    _arrow(fig, mon_x, y_cnt, mon_x, BOT_Y1, mcol, dash=True, width=1.4)
    solid = dict(color=ccol, width=1.6 if rec else 1.2)
    fig.add_shape(type="line", x0=rc_x, y0=BOT_Y1, x1=rc_x, y1=y_cmd, line=solid)
    fig.add_shape(type="line", x0=taps[0] + 3, y0=y_cmd, x1=rc_x, y1=y_cmd, line=solid)
    for x in taps:
        _arrow(fig, x + 3, y_cmd, x + 3, TOP_Y0, ccol, width=1.6 if rec else 1.2)
    _text(fig, mon_x - 1, y_cnt + 1.4, "performance counters", 9, mcol, anchor="left")
    _text(fig, rc_x + 1, y_cmd - 1.4, "reconfiguration commands", 9, ccol, anchor="right")

    # activity legend (top), evenly spaced so it never overflows
    labels = [("normal", True), ("monitoring", activity in ("monitoring", "deciding")),
              ("deciding", activity == "deciding"), ("reconfiguring", rec is not None), ("overhead", overhead)]
    for i, (key, on) in enumerate(labels):
        lx = 3.0 + i * 19.5
        color = T.ACTIVITY_COLORS[key]
        fig.add_shape(type="circle", x0=lx, y0=55.3, x1=lx + 1.3, y1=57.0,
                      line=dict(color=color, width=1.5), fillcolor=color if on else T.SURFACE)
        _text(fig, lx + 1.9, 56.15, T.ACTIVITY_LABELS[key], 10, T.INK if on else T.MUTED, anchor="left")

    fig.update_xaxes(visible=False, range=[0, W], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, H], fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor=T.SURFACE,
                      plot_bgcolor=T.SURFACE, showlegend=False, dragmode=False)
    return fig


def _short(field: str) -> str:
    return {"cores": "cores", "freq_ghz": "freq", "cache_kb": "LLC", "mem_channels": "channels",
            "prefetch": "prefetch", "policy": "policy"}[field]
