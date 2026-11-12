"""
Self-Evolving Computer Architecture - interactive dashboard.

Run with:  streamlit run app.py

The dashboard only *drives and observes* the simulation engines in
``simulation/``; every number shown is read from engine state.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from config import default_config as K
from simulation.analysis import compare, phase_length_sweep, run_comparison, short_workload_sweep
from simulation.baseline_architecture import create_adaptive_engine, create_static_engine
from simulation.models import ArchConfig, SimulationParams, fmt_cache
from simulation.scenarios import CUSTOM_KEY, SCENARIOS
from simulation.workload_generator import PHASE_KINDS, PhaseSpec
from visualization import metrics_dashboard as MD
from visualization import performance_charts as PC
from visualization import theme as T
from visualization.architecture_diagram import build_architecture_figure

st.set_page_config(page_title="Self-Evolving Architecture Simulator", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(MD.CSS, unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False}   # hover tooltips stay; toolbar would cover legends
KIND_LABEL = {"compute": "Compute-intensive", "memory": "Memory-intensive", "mixed": "Mixed"}

# =============================================================================
# Sidebar: configuration
# =============================================================================
with st.sidebar:
    st.markdown("### Scenario")
    scen_keys = list(SCENARIOS) + [CUSTOM_KEY]
    scen_key = st.selectbox("Workload scenario", scen_keys,
                            format_func=lambda k: SCENARIOS[k].title if k in SCENARIOS else "7. Custom workload",
                            index=2)
    if scen_key in SCENARIOS:
        st.caption(SCENARIOS[scen_key].purpose)
        base_phases = SCENARIOS[scen_key].phases
    else:
        st.caption("Define your own phase sequence below.")
        base_phases = SCENARIOS["dynamic"].phases

    editable = scen_key == CUSTOM_KEY
    phase_df = pd.DataFrame([{"Phase kind": p.kind, "Work (M instructions)": round(p.instructions / 1e6, 3)}
                             for p in base_phases])
    with st.expander("Workload phases", expanded=editable):
        if editable:
            phase_df = st.data_editor(
                phase_df, num_rows="dynamic", width="stretch", key="phase_editor",
                column_config={
                    "Phase kind": st.column_config.SelectboxColumn(options=list(PHASE_KINDS), required=True),
                    "Work (M instructions)": st.column_config.NumberColumn(min_value=0.01, max_value=500.0,
                                                                           step=0.1, required=True),
                })
            st.caption("~100 static-baseline steps ≈ compute 22 M · memory 2.4 M · mixed 8 M instructions.")
        else:
            st.dataframe(phase_df, width="stretch", hide_index=True)
            st.caption("Choose *Custom workload* to edit phases.")

    st.markdown("### Monitoring & decisions")
    sampling = st.slider("Sampling interval (steps)", 1, 20, K.DEFAULT_SAMPLING_INTERVAL)
    confirm = st.slider("Consecutive samples to confirm a class", 1, 5, K.DEFAULT_CONFIRM_SAMPLES)
    min_gain = st.slider("Min. predicted gain to reconfigure (%)", 0, 50, int(K.DEFAULT_MIN_GAIN * 100))
    margin = st.slider("Benefit / cost safety margin (×)", 0.5, 5.0, K.DEFAULT_BENEFIT_MARGIN, 0.25)
    power = st.slider("Power budget (W)", 25, 80, int(K.DEFAULT_POWER_BUDGET_W))
    cost_aware = st.toggle("Cost-aware decisions", True,
                           help="Off = reconfigure whenever a better config is predicted, ignoring cost")
    hysteresis = st.toggle("Classifier hysteresis", True)

    with st.expander("Simulation engine"):
        step_cycles = st.select_slider("Cycles per step", [2_000, 5_000, 10_000, 20_000, 50_000],
                                       K.DEFAULT_STEP_CYCLES)
        noise = st.slider("Workload noise (σ, %)", 0, 20, 0,
                          help="Optional seeded perturbation of phase characteristics; identical for both systems.")
        seed = st.number_input("Noise seed", 0, 10_000, 7)
        expected_phase = st.slider("Prior expected phase length (steps)", 5, 200, int(K.DEFAULT_EXPECTED_PHASE_STEPS))

    with st.expander("Static baseline / initial configuration"):
        sc = K.DEFAULT_STATIC_CONFIG
        s_cores = st.select_slider("Cores", K.CORE_OPTIONS, sc["cores"])
        s_freq = st.select_slider("Frequency (GHz)", K.FREQ_OPTIONS_GHZ, sc["freq_ghz"])
        s_cache = st.select_slider("LLC capacity", K.CACHE_OPTIONS_KB, sc["cache_kb"], format_func=fmt_cache)
        s_ch = st.select_slider("Memory channels", K.CHANNEL_OPTIONS, sc["mem_channels"])
        s_pf = st.toggle("Prefetcher", sc["prefetch"])
        s_pol = st.radio("Replacement policy", K.REPLACEMENT_POLICIES, horizontal=True,
                         index=K.REPLACEMENT_POLICIES.index(sc["policy"]))

    st.markdown("### Playback")
    speed = st.slider("Simulation speed (steps per refresh)", 1, 25, 3)
    refresh = st.select_slider("Refresh interval (s)", [0.25, 0.5, 0.75, 1.0, 1.5], 0.5)

# -----------------------------------------------------------------------------
static_cfg = ArchConfig(int(s_cores), float(s_freq), int(s_cache), int(s_ch), bool(s_pf), str(s_pol))
phases = [PhaseSpec(str(r["Phase kind"]), float(r["Work (M instructions)"]) * 1e6)
          for _, r in phase_df.dropna().iterrows() if float(r["Work (M instructions)"]) > 0]
params = SimulationParams(
    step_cycles=int(step_cycles), sampling_interval=int(sampling), confirm_samples=int(confirm),
    min_gain=min_gain / 100.0, benefit_margin=float(margin), expected_phase_steps=float(expected_phase),
    power_budget_w=float(power), cost_aware=cost_aware, hysteresis=hysteresis, noise=noise / 100.0,
    seed=int(seed), initial_config=static_cfg,
)
signature = json.dumps({"p": asdict(params), "ph": [(p.kind, p.instructions) for p in phases]},
                       sort_keys=True, default=str)


def reset_engines():
    st.session_state.adaptive = create_adaptive_engine(phases, params)
    st.session_state.static = create_static_engine(phases, params, static_cfg)
    st.session_state.signature = signature
    st.session_state.running = False


if not phases:
    st.error("The workload needs at least one phase with positive work.")
    st.stop()
if st.session_state.get("signature") != signature:
    reset_engines()


def advance(n: int):
    a, s = st.session_state.adaptive, st.session_state.static
    for _ in range(n):
        if a.finished and s.finished:
            break
        a.step()
        s.step()


# =============================================================================
# Header & controls
# =============================================================================
st.markdown("## 🧬 Self-Evolving Computer Architecture")
st.caption("Runtime monitoring → workload characterisation → autonomous decisions → dynamic reconfiguration, "
           "simulated against a conventional static architecture running the identical workload.")

b = st.columns([1, 1, 1, 1, 1.2, 1, 3])
if b[0].button("▶ Start", width="stretch", type="primary"):
    if st.session_state.adaptive.finished and st.session_state.static.finished:
        reset_engines()
    st.session_state.running = True
if b[1].button("⏸ Pause", width="stretch"):
    st.session_state.running = False
if b[2].button("⏯ Resume", width="stretch"):
    st.session_state.running = True
if b[3].button("⏭ Step", width="stretch"):
    st.session_state.running = False
    advance(1)
if b[4].button("⏩ Run to end", width="stretch"):
    st.session_state.running = False
    advance(K.MAX_STEPS)
if b[5].button("↺ Reset", width="stretch"):
    reset_engines()


# =============================================================================
# Live area (re-runs on a timer while the simulation is playing)
# =============================================================================
def render():
    a, s = st.session_state.adaptive, st.session_state.static
    running = st.session_state.running
    state = ("Running" if running else "Paused") if a.history else "Ready"
    if a.finished and s.finished:
        state = "Finished"
    chip = lambda t: f'<span class="status-chip">{t}</span>'
    st.markdown(
        chip(f"● {state}") + chip(f"Adaptive: step {a.step_index} · cycle {a.cycle:,.0f}"
                                  + (" ✓" if a.finished else "")) +
        chip(f"Static: step {s.step_index} · cycle {s.cycle:,.0f}" + (" ✓" if s.finished else "")) +
        chip(f"{len(a.controller.history)} reconfigurations"),
        unsafe_allow_html=True)

    tabs = st.tabs(["🏗 Architecture", "📈 Performance", "⚖ Static vs Adaptive", "⏱ Adaptation overhead",
                    "📜 Event log", "∑ Model & equations"])

    # ---------------------------------------------------------------- architecture
    with tabs[0]:
        st.plotly_chart(build_architecture_figure(a), width="stretch", config={"staticPlot": True},
                        key="arch_fig")
        st.markdown(MD.live_metric_cards(a), unsafe_allow_html=True)
        st.markdown("**Latest events**")
        st.markdown(MD.event_log_html(a.events, limit=10), unsafe_allow_html=True)

    # ---------------------------------------------------------------- performance
    with tabs[1]:
        st.caption("Blue = adaptive, orange = static. Amber bands = reconfiguration in progress; "
                   "dotted lines = actual workload phase transitions.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(PC.timeseries(a, s, "cpu_util", "CPU utilisation", "%", pct=True),
                        width="stretch", config=PLOT_CFG, key="ts_cpu")
        c2.plotly_chart(PC.timeseries(a, s, "mem_bw_util", "Memory bandwidth utilisation", "%", pct=True),
                        width="stretch", config=PLOT_CFG, key="ts_bw")
        c1, c2 = st.columns(2)
        c1.plotly_chart(PC.timeseries(a, s, "ipc", "IPC (instructions per core-clock, all cores)", "IPC"),
                        width="stretch", config=PLOT_CFG, key="ts_ipc")
        c2.plotly_chart(PC.timeseries(a, s, "hit_rate", "LLC hit rate", "%", pct=True),
                        width="stretch", config=PLOT_CFG, key="ts_hit")
        c1, c2 = st.columns(2)
        c1.plotly_chart(PC.classification_chart(a), width="stretch", config=PLOT_CFG, key="ts_cls")
        c2.plotly_chart(PC.timeseries(a, s, "mem_latency_ns", "Memory access latency", "ns"),
                        width="stretch", config=PLOT_CFG, key="ts_lat")
        st.markdown("**Architecture configuration over time**")
        st.plotly_chart(PC.config_chart(a, s), width="stretch", config=PLOT_CFG, key="cfg_chart")

    # ---------------------------------------------------------------- comparison
    with tabs[2]:
        both_done = a.finished and s.finished
        if not both_done:
            st.info("Live, partial comparison — results become final when both architectures finish the "
                    "workload (use **Run to end** for an instant result).")
        try:
            cmp = compare(s, a) if both_done else None
        except ValueError as e:
            st.error(str(e))
            cmp = None
        sa, ss = a.summary(), s.summary()
        if cmp:
            imp = cmp["improvement_pct"]
            col = T.GOOD if imp > 0 else T.CRITICAL
            h = st.columns(4)
            h[0].markdown(MD.hero("Performance improvement", f"{imp:+.1f}%",
                                  f"{'faster' if imp > 0 else 'slower'} than static · speed-up ×{cmp['speedup']:.2f}",
                                  col), unsafe_allow_html=True)
            h[1].markdown(MD.hero("Net benefit", f"{cmp['net_benefit_cycles'] / 1e3:,.0f} k cyc",
                                  f"gross {cmp['gross_benefit_cycles'] / 1e3:,.0f} k − overhead "
                                  f"{sa['adaptation_overhead_cycles'] / 1e3:,.0f} k"), unsafe_allow_html=True)
            h[2].markdown(MD.hero("Adaptation overhead", f"{sa['adaptation_overhead_pct']:.2f}%",
                                  f"{sa['reconfig_count']} reconfigurations · "
                                  f"{sa['reconfig_cycles'] / 1e3:,.0f} k cycles in transition"),
                          unsafe_allow_html=True)
            h[3].markdown(MD.hero("Energy", f"{cmp['energy_change_pct']:+.1f}%",
                                  f"energy-delay product {cmp['edp_change_pct']:+.1f}%"), unsafe_allow_html=True)
            st.write("")
            st.plotly_chart(PC.comparison_bars(cmp), width="stretch", config=PLOT_CFG, key="cmp_bars")
        st.plotly_chart(PC.progress_chart(a, s), width="stretch", config=PLOT_CFG, key="progress")

        rows = [
            ("Total execution cycles", "total_cycles", "{:,.0f}"),
            ("Normal execution cycles", "normal_execution_cycles", "{:,.0f}"),
            ("Adaptation overhead cycles", "adaptation_overhead_cycles", "{:,.0f}"),
            ("Average IPC", "avg_ipc", "{:.3f}"),
            ("CPU utilisation", "avg_cpu_util", "{:.1%}"),
            ("Memory bandwidth utilisation", "avg_mem_bw_util", "{:.1%}"),
            ("LLC hit rate", "avg_hit_rate", "{:.1%}"),
            ("Average memory latency (ns)", "avg_mem_latency_ns", "{:.1f}"),
            ("Resource utilisation", "avg_resource_util", "{:.1%}"),
            ("Average power (W)", "avg_power_w", "{:.1f}"),
            ("Energy (µJ)", "energy_uj", "{:,.0f}"),
            ("Reconfigurations", "reconfig_count", "{:d}"),
            ("Instructions retired", "instructions", "{:,.0f}"),
        ]
        st.dataframe(pd.DataFrame([{"Metric": n, "Static": f.format(ss[k]), "Adaptive": f.format(sa[k])}
                                   for n, k, f in rows]), width="stretch", hide_index=True)
        st.caption("Total effective execution time = normal execution time + adaptation overhead. "
                   "Improvement = (T_static − T_adaptive) / T_static × 100.")

    # ---------------------------------------------------------------- overhead
    with tabs[3]:
        st.markdown("#### This run")
        c1, c2 = st.columns([2, 3])
        c1.markdown(MD.kpi_grid([
            MD.kpi("Reconfigurations", f"{sa['reconfig_count']}", f"{sa['holds']} decisions to hold"),
            MD.kpi("Cycles in transition", f"{sa['reconfig_cycles']:,.0f}", "start → completion, summed"),
            MD.kpi("Work lost to reconfig", f"{sa['reconfig_lost_cycles']:,.0f}", "cycle-equivalents"),
            MD.kpi("Monitoring", f"{sa['monitor_overhead_cycles']:,.0f}", "cycles (PMU samples)"),
            MD.kpi("Decision-making", f"{sa['decision_overhead_cycles']:,.0f}", "cycles (optimiser runs)"),
            MD.kpi("Reconfig energy", f"{sa['reconfig_energy_uj']:,.1f} µJ", "PLL, power gates, flushes"),
            MD.kpi("Overhead share", f"{sa['adaptation_overhead_pct']:.2f}%", "of total execution time"),
        ]), unsafe_allow_html=True)
        c2.plotly_chart(PC.overhead_breakdown(sa), width="stretch", config=PLOT_CFG, key="ovh")
        st.markdown("**Reconfiguration records**")
        rdf = MD.reconfig_df(a)
        if rdf.empty:
            st.caption("No reconfiguration yet.")
        else:
            st.dataframe(rdf, width="stretch", hide_index=True)
        st.markdown("**Decision engine log** (every time the optimiser ran)")
        ddf = MD.decisions_df(a)
        if ddf.empty:
            st.caption("The decision engine has not run yet.")
        else:
            st.dataframe(ddf, width="stretch", hide_index=True)

        st.markdown("#### When is adaptation *not* worth it?")
        st.caption("Batch experiments on the same engine. They run the full simulation many times, "
                   "so they are computed on request.")
        if st.button("Run overhead experiments", key="run_sweeps"):
            st.session_state.sweeps = _sweeps(signature)
        if "sweeps" in st.session_state and st.session_state.sweeps.get("sig") == signature:
            sw = st.session_state.sweeps
            c1, c2 = st.columns(2)
            c1.plotly_chart(PC.sweep_chart(sw["phase"], "phase_length_steps", "phase length (static-baseline steps, log)",
                                           "Frequent phase switching (fixed total work)", log_x=True),
                            width="stretch", config=PLOT_CFG, key="sw1")
            c2.plotly_chart(PC.sweep_chart(sw["short"], "length_steps", "workload length (static-baseline steps, log)",
                                           "Short workloads",
                                           series_key=None, log_x=True),
                            width="stretch", config=PLOT_CFG, key="sw2")
            st.markdown("**Cost-aware vs naive adaptation on the current workload**")
            st.dataframe(pd.DataFrame(sw["naive_vs_smart"]), width="stretch", hide_index=True)
            st.caption("Naive = samples every step, accepts the first sample of a new class, no hysteresis "
                       "and no cost/benefit check. It reacts faster on long phases but thrashes when phases "
                       "are short.")

    # ---------------------------------------------------------------- events
    with tabs[4]:
        all_cats = ["init", "phase", "monitor", "classify", "decision", "reconfig", "hold", "done"]
        hidden = st.multiselect("Hide event categories", all_cats, default=[], key="ev_hide")
        shown = set(all_cats) - set(hidden)
        st.caption(f"{sum(e.category in shown for e in a.events)} of {len(a.events)} events shown")
        st.markdown(MD.event_log_html(a.events, categories=shown).replace("max-height: 300px", "max-height: 600px"),
                    unsafe_allow_html=True)
        st.download_button("Download event log (CSV)", MD.events_df(a.events).to_csv(index=False),
                           file_name="event_log.csv", mime="text/csv", key="dl_events")

    # ---------------------------------------------------------------- model
    with tabs[5]:
        _model_tab()


@st.cache_data(show_spinner="Running batch simulations…")
def _sweeps(sig: str) -> dict:
    p = params
    out = {"sig": sig, "phase": phase_length_sweep(p), "short": short_workload_sweep(p)}
    rows = []
    for label, naive in (("Cost-aware adaptive", False), ("Naive adaptive", True)):
        r = run_comparison(phases, p, static_cfg, naive=naive)
        rows.append({"Policy": label, "Improvement vs static": f"{r['improvement_pct']:+.1f}%",
                     "Reconfigurations": r["adaptive"]["reconfig_count"],
                     "Overhead": f"{r['adaptive']['adaptation_overhead_pct']:.1f}%",
                     "Energy vs static": f"{r['energy_change_pct']:+.1f}%"})
    out["naive_vs_smart"] = rows
    return out


def _model_tab():
    st.markdown(r"""
**Time base.** 1 reference cycle = 1 ns (1 GHz reference clock). One engine step = *cycles per step* reference cycles.
Phases are defined by **work** (instructions), so a faster machine finishes them sooner.

**CPU.** Amdahl speed-up $S(n) = \dfrac{1}{(1-p) + p/n}$. Per-thread time per instruction
$t = t_{exec} + t_{mem}$ with
$t_{exec} = \dfrac{CPI_{exec} + A_{llc}(1-m)\,H(C)\,e_{hit}}{f}$, $\;H(C) = 10 + 3\log_2(C/256\,\mathrm{KB})$ cycles.

**Cache.** Coverage $cov=\min(1, R/W)$ of the working set $W$ by resident data $R$;
$m = m_{min} + (m_{base}-m_{min})(1-\sqrt{cov})$ (√2 rule). FIFO: $m \times (1 + 0.15(L_t-0.4))$.
Prefetcher: coverage $c = 0.8s$, accuracy $a = 0.5+0.4s$, exposed misses $m(1-c)$, extra traffic $mc(1/a-1)$.
Resident data warms as misses install lines and is mostly invalidated at a phase change.

**Memory.** $B_{peak} = 12.8\,k$ GB/s, $\rho = D/B_{peak}$, $L(\rho) = 75 + 30\,\rho/(1-\rho)$ ns,
queue occupancy by Little's law.

**Coupling.** Throughput $X$ is the fixed point of $X = S(n) / \big(t_{exec} + A_{llc}\,m_{exp}\,L(X b / B_{peak})/MLP\big)$
(solved by bisection, capped at 97 % bus utilisation). IPC $= X/f$, CPU utilisation $=(S/n)\,t_{exec}/t$.

**Power.** $P = n\,[0.8 + 1.6 f V^2 (0.35 + 0.65u)] + 1.0\,C_{MB} + k(1.5 + 2.5\rho) + 0.4\,PF$, $V = 0.6 + 0.15f$.
Candidates must fit the power budget at peak.

**Decision.** Counter-based predictor (Amdahl inverse from busy-core ratio, MLP from the CPI stack, √2 miss scaling).
Reconfigure iff gain ≥ threshold **and** $H \cdot \text{step} \cdot (1 - X_0/X_1) \ge \text{margin} \times \text{lost cycles}$,
$H$ = learnt mean phase residency.

**Reconfiguration.** duration $= 2000 + \max_i \text{latency}_i + \text{flush}$; throughput × $\min_i$ factor$_i$ during the
transition on the resources common to old and new configuration. Lost cycles are measured against an ideal
*instantaneous* switch: $\sum_{\text{transition}} \Delta t\,(1 - \text{factor}\cdot X_{trans}/X_{new})$.

See `README.md` for the full derivation and assumptions.
""")
    st.markdown("**Reconfiguration cost table** (from `config/default_config.py`)")
    st.dataframe(pd.DataFrame([{"Resource": k, "Latency (cycles)": v["latency"], "Throughput factor": v["factor"],
                                "Energy (µJ)": v["energy_uj"]} for k, v in K.RECONFIG_COSTS.items()]),
                 width="stretch", hide_index=True)


run_every = refresh if st.session_state.running else None


@st.fragment(run_every=run_every)
def live():
    if st.session_state.running:
        advance(speed)
        a, s = st.session_state.adaptive, st.session_state.static
        if a.finished and s.finished:
            st.session_state.running = False
            st.rerun()
    render()


live()
