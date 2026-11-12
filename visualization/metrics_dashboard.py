"""Live metric cards and tables rendered as HTML for Streamlit."""

from __future__ import annotations

import html

import pandas as pd

from simulation.models import fmt_cache
from visualization import theme as T

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin: 4px 0 14px; }}
.kpi {{ background: {T.SURFACE}; border: 1px solid {T.BORDER}; border-radius: 8px; padding: 10px 12px; }}
.kpi .lbl {{ color: {T.MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .val {{ color: {T.INK}; font-size: 1.35rem; font-weight: 600; margin-top: 2px; line-height: 1.2; }}
.kpi .sub {{ color: {T.INK_2}; font-size: 0.75rem; margin-top: 2px; }}
.kpi .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align: middle; }}
.hero {{ background: {T.SURFACE}; border: 1px solid {T.BORDER}; border-radius: 10px; padding: 14px 16px; }}
.hero .val {{ font-size: 2.1rem; font-weight: 700; color: {T.INK}; }}
.hero .lbl {{ color: {T.MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; }}
.hero .sub {{ color: {T.INK_2}; font-size: 0.8rem; }}
.status-chip {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
               border: 1px solid {T.BORDER}; color: {T.INK}; background: {T.SURFACE_2}; margin-right: 6px; }}
.evlog {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; line-height: 1.55;
          background: {T.SURFACE}; border: 1px solid {T.BORDER}; border-radius: 8px; padding: 10px 12px;
          max-height: 300px; overflow-y: auto; }}
.evlog .cyc {{ color: {T.MUTED}; }}
.evlog .cat {{ font-weight: 600; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
</style>
"""

EVENT_COLORS = {
    "init": T.MUTED, "phase": T.KIND_COLORS["mixed"], "monitor": T.ADAPTIVE, "classify": T.ADAPTIVE,
    "decision": T.NAIVE, "hold": T.INK_2, "reconfig": T.WARNING, "done": T.GOOD,
}


def kpi(label: str, value: str, sub: str = "", dot: str | None = None) -> str:
    d = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
    return (f'<div class="kpi"><div class="lbl">{d}{html.escape(label)}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def kpi_grid(cards: list[str]) -> str:
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


def hero(label: str, value: str, sub: str, color: str = T.INK) -> str:
    return (f'<div class="hero"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>')


def live_metric_cards(engine) -> str:
    last = engine.history[-1] if engine.history else None
    cfg = engine.config
    cls = engine.classifier.stable if engine.adaptive else "n/a (static)"
    if last is None:
        return kpi_grid([kpi("Status", "Ready", "press Start or Step")])
    act = "reconfiguring" if engine.controller.busy else last.activity
    cards = [
        kpi("Workload class", cls, f"true phase: {engine.workload.current_phase.label}",
            T.CLASS_COLORS.get(cls, T.MUTED)),
        kpi("CPU utilisation", f"{last.cpu_util:.0%}", "active (non-stalled) core time"),
        kpi("IPC", f"{last.ipc:.2f}", f"{last.ipc / max(cfg.cores, 1):.2f} per core"),
        kpi("LLC hit rate", f"{last.hit_rate:.0%}", f"warm {engine.cache.resident_kb / 1024:.2f} MB"),
        kpi("Memory BW util", f"{last.mem_bw_util:.0%}", f"{last.mem_bw_gbps:.1f} GB/s"),
        kpi("Memory latency", f"{last.mem_latency_ns:.0f} ns", f"queue {last.queue_util:.0%}"),
        kpi("Active cores", f"{cfg.cores} / 8", "CPU core allocation"),
        kpi("CPU frequency", f"{cfg.freq_ghz:.1f} GHz", "DVFS"),
        kpi("Cache capacity", fmt_cache(cfg.cache_kb), f"{cfg.policy} replacement"),
        kpi("Memory channels", f"{cfg.mem_channels} / 4", f"peak {cfg.mem_channels * 12.8:.1f} GB/s"),
        kpi("Prefetcher", "Enabled" if cfg.prefetch else "Disabled", "stream prefetch"),
        kpi("Activity", T.ACTIVITY_LABELS.get(act, act), f"power {last.power_w:.1f} W", T.ACTIVITY_COLORS.get(act)),
    ]
    return kpi_grid(cards)


def event_log_html(events, limit: int | None = None, categories: set | None = None) -> str:
    rows = [e for e in events if categories is None or e.category in categories]
    if limit:
        rows = rows[-limit:]
    lines = []
    for e in rows:
        c = EVENT_COLORS.get(e.category, T.INK_2)
        lines.append(f'<div><span class="cyc">Cycle {e.cycle:>12,.0f}</span> '
                     f'<span class="cat" style="color:{c}">[{e.category}]</span> {html.escape(e.message)}</div>')
    return '<div class="evlog">' + ("".join(lines) or "No events yet.") + "</div>"


def events_df(events) -> pd.DataFrame:
    return pd.DataFrame([{"Cycle": int(round(e.cycle)), "Category": e.category, "Event": e.message} for e in events])


def reconfig_df(engine) -> pd.DataFrame:
    rows = []
    for r in engine.controller.history:
        rows.append({
            "#": r.id,
            "Start cycle": int(r.start_cycle),
            "End cycle": int(r.end_cycle),
            "Duration (cycles)": int(r.duration),
            "Previous configuration": r.old.describe(),
            "New configuration": r.new.describe(),
            "Changed": ", ".join(r.changed),
            "Throughput during": f"×{r.throughput_factor:.2f}",
            "Lost cycles": int(r.lost_cycles),
            "Energy (µJ)": round(r.energy_uj, 1),
            "Predicted gain": f"{r.predicted_gain:+.1%}",
            "Status": "done" if r.completed else "in progress",
        })
    return pd.DataFrame(rows)


def decisions_df(engine) -> pd.DataFrame:
    return pd.DataFrame([{
        "Cycle": int(d.cycle), "Class": d.workload_class, "Action": d.action,
        "Predicted gain": f"{d.predicted_gain:+.1%}", "Horizon (steps)": round(d.horizon_steps, 1),
        "Expected saving (cycles)": int(d.expected_saved_cycles),
        "Expected cost (cycles)": int(d.expected_cost_cycles),
        "Candidates": d.candidates_evaluated, "Reason": d.reason,
    } for d in engine.decider.decisions])
