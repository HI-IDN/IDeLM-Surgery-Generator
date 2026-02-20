"""Interactive visual summary of generated surgery data.

Run with:
    python -m src.visualize

Produces a self-contained ``surgery_data_dashboard.html`` in the project root
and opens it automatically in your browser.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .generate_all_data import generate_all_data
from .generators import params as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SECTION_STYLE = (
    "font-family:sans-serif;padding:8px 16px;margin-top:32px;"
    "background:#1f2937;color:#f9fafb;border-radius:6px;"
)


def _html_section(title: str, fig: go.Figure) -> str:
    """Return an HTML string: a heading followed by the Plotly figure."""
    heading = f'<h2 style="{SECTION_STYLE}">{title}</h2>'
    fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return heading + fig_html


# ---------------------------------------------------------------------------
# Individual plot builders
# ---------------------------------------------------------------------------


def plot_waiting_list(waiting_list) -> go.Figure:
    """2 x 3 panel: key statistics for every surgery on the waiting list."""
    durations = [s.expected_duration for s in waiting_list]
    operate_by = [s.operate_by for s in waiting_list]
    days_reg = [s.days_since_registration for s in waiting_list]
    icu_flags = [s.icu for s in waiting_list]
    ward_flags = [s.ward for s in waiting_list]
    los_icu = [s.los_icu for s in waiting_list if s.icu]
    los_ward = [s.los_ward for s in waiting_list if s.ward]
    allowed_changes = [s.allowed_changes for s in waiting_list]

    icu_only = sum(i and not w for i, w in zip(icu_flags, ward_flags))
    ward_only = sum(w and not i for i, w in zip(icu_flags, ward_flags))
    both = sum(i and w for i, w in zip(icu_flags, ward_flags))
    neither = sum(not i and not w for i, w in zip(icu_flags, ward_flags))

    hover_dur = [
        f"Surgery {s.id}<br>Op: {s.operation_card_id}<br>Surgeon: {s.surgeon_id}"
        f"<br>Duration: {s.expected_duration} min"
        for s in waiting_list
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Expected Duration (min)",
            "Operate-By Day",
            "Post-Op Care Mix",
            "ICU Length of Stay (days)",
            "Ward Length of Stay (days)",
            "Registration Days vs. Duration",
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # 1 – Duration histogram
    fig.add_trace(
        go.Histogram(
            x=durations,
            nbinsx=30,
            name="Duration",
            marker_color="#6366f1",
            hovertemplate="Duration: %{x} min<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2 – Operate-by histogram
    fig.add_trace(
        go.Histogram(
            x=operate_by,
            nbinsx=30,
            name="Operate-By",
            marker_color="#10b981",
            hovertemplate="By day %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3 – Post-op care pie
    fig.add_trace(
        go.Pie(
            labels=["ICU only", "Ward only", "ICU + Ward", "Neither"],
            values=[icu_only, ward_only, both, neither],
            hole=0.35,
            marker_colors=["#ef4444", "#3b82f6", "#f59e0b", "#6b7280"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} surgeries (%{percent})<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # 4 – ICU LOS histogram
    if los_icu:
        fig.add_trace(
            go.Histogram(
                x=los_icu,
                nbinsx=20,
                name="ICU LOS",
                marker_color="#ef4444",
                hovertemplate="LOS: %{x} days<br>Count: %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No ICU cases", row=2, col=1, showarrow=False, font_size=14
        )

    # 5 – Ward LOS histogram
    if los_ward:
        fig.add_trace(
            go.Histogram(
                x=los_ward,
                nbinsx=20,
                name="Ward LOS",
                marker_color="#3b82f6",
                hovertemplate="LOS: %{x} days<br>Count: %{y}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    else:
        fig.add_annotation(
            text="No Ward cases", row=2, col=2, showarrow=False, font_size=14
        )

    # 6 – Registration days vs duration scatter, coloured by allowed_changes
    fig.add_trace(
        go.Scatter(
            x=days_reg,
            y=durations,
            mode="markers",
            marker=dict(
                color=allowed_changes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Allowed<br>Changes", x=1.02),
                size=6,
                opacity=0.7,
            ),
            text=hover_dur,
            hovertemplate="%{text}<br>Days since reg: %{x}<extra></extra>",
            name="Surgeries",
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        height=720,
        showlegend=False,
        template="plotly_dark",
        title_text="Waiting List – At a Glance",
        title_font_size=18,
    )
    fig.update_xaxes(title_text="Minutes", row=1, col=1)
    fig.update_xaxes(title_text="Day", row=1, col=2)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_xaxes(title_text="Days", row=2, col=2)
    fig.update_xaxes(title_text="Days since registration", row=2, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Expected duration (min)", row=2, col=3)
    return fig


def plot_case_mix(frequency_data, top_n: int = 20) -> go.Figure:
    """Bar + heatmap: how operation-card frequencies are distributed."""
    from collections import defaultdict

    card_totals: dict[str, float] = defaultdict(float)
    for (card, _), freq in frequency_data.items():
        card_totals[card] += freq

    sorted_cards = sorted(card_totals, key=lambda c: -card_totals[c])
    top_cards = sorted_cards[:top_n]

    surgeons = sorted({s for (_, s) in frequency_data.keys()})

    # Heatmap matrix
    z = [[frequency_data.get((card, s), 0.0) for s in surgeons] for card in top_cards]
    hover_hm = [
        [
            f"Op: {card}<br>Surgeon: {s}<br>Frequency: {frequency_data.get((card, s), 0.0):.5f}"
            for s in surgeons
        ]
        for card in top_cards
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Top {top_n} Operation Cards by Total Frequency",
            f"Frequency Heatmap – Top {top_n} Cards × Surgeons",
        ],
        column_widths=[0.35, 0.65],
        horizontal_spacing=0.10,
    )

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=[card_totals[c] for c in top_cards],
            y=top_cards,
            orientation="h",
            marker_color="#818cf8",
            hovertemplate="<b>%{y}</b><br>Frequency: %{x:.5f}<extra></extra>",
            name="Total freq",
        ),
        row=1,
        col=1,
    )

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=[f"S{s}" for s in surgeons],
            y=top_cards,
            colorscale="Plasma",
            hovertext=hover_hm,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(title="Freq", x=1.01),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        showlegend=False,
        title_text="Case Mix & Surgeon–Card Frequencies",
        title_font_size=18,
    )
    fig.update_xaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    return fig


def plot_schedule(schedule, n_surgeons: int) -> go.Figure:
    """Heatmap: surgeon × (day–room) schedule fractions in a single view."""
    surgeons = sorted({s for (s, _, _) in schedule.keys()})
    rooms = sorted({r for (_, r, _) in schedule.keys()})
    days = sorted({d for (_, _, d) in schedule.keys()})

    # Column labels ordered day-first: Mon-R0, Mon-R1, …, Fri-R4
    col_labels = [f"{DAY_NAMES[d]}-R{r}" for d in days for r in rooms]
    col_keys = [(d, r) for d in days for r in rooms]

    z = [[schedule.get((s, r, d), 0.0) for (d, r) in col_keys] for s in surgeons]
    hover = [
        [
            f"Surgeon {s}<br>{DAY_NAMES[d]} · Room {r}"
            f"<br>Schedule fraction: {schedule.get((s, r, d), 0.0):.4f}"
            for (d, r) in col_keys
        ]
        for s in surgeons
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=[f"Surgeon {s}" for s in surgeons],
            colorscale="Teal",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(title="Fraction"),
        )
    )

    # Vertical lines between days to separate them visually
    n_rooms = len(rooms)
    for i in range(1, len(days)):
        fig.add_vline(
            x=i * n_rooms - 0.5,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
        )

    fig.update_layout(
        height=max(350, 50 * len(surgeons) + 120),
        template="plotly_dark",
        title_text="Weekly Schedule – Surgeon × Day/Room Fraction",
        title_font_size=18,
        xaxis=dict(title="Day – Room", tickangle=-45),
        yaxis=dict(title="Surgeon", autorange="reversed"),
        margin=dict(b=100),
    )
    return fig


def plot_duration_data(duration_data, top_n: int = 20) -> go.Figure:
    """Expected-duration box plots per op-card + kappa distribution."""
    from collections import defaultdict

    # Aggregate expected durations per operation card
    card_durations: dict[str, list[float]] = defaultdict(list)
    kappas: list[float] = []
    for (card, _), cell in duration_data.items():
        mu, sigma, gamma = cell["mu"], cell["sigma"], cell["gamma"]
        expected = gamma + np.exp(mu + 0.5 * sigma**2)
        card_durations[card].append(expected)
        kappas.append(cell["kappa"])

    # Pick top-N cards by median duration
    sorted_cards = sorted(
        card_durations, key=lambda c: -float(np.median(card_durations[c]))
    )[:top_n]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Expected Duration Distribution – Top {top_n} Cards (by median)",
            "Surgeon Speed Multiplier (κ) Distribution",
        ],
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.10,
    )

    colors = [f"hsl({int(i * 360 / top_n)},70%,55%)" for i in range(top_n)]

    for i, card in enumerate(sorted_cards):
        vals = card_durations[card]
        fig.add_trace(
            go.Box(
                y=vals,
                name=card,
                marker_color=colors[i],
                hovertemplate=(
                    f"<b>{card}</b><br>Expected duration: %{{y:.1f}} min<extra></extra>"
                ),
                boxpoints="outliers",
                jitter=0.3,
                whiskerwidth=0.6,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=kappas,
            nbinsx=30,
            marker_color="#f97316",
            hovertemplate="κ = %{x:.3f}<br>Count: %{y}<extra></extra>",
            name="κ",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        showlegend=False,
        title_text="Duration Parameters",
        title_font_size=18,
    )
    fig.update_yaxes(title_text="Expected duration (min)", row=1, col=1)
    fig.update_xaxes(title_text="κ (speed multiplier)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    return fig


def plot_priority_admission(priority_data, admission_data) -> go.Figure:
    """Priority levels and admission probabilities across operation cards."""
    cards = list(priority_data.keys())
    operate_by = [priority_data[c]["operate_by"] for c in cards]
    allowed_changes = [priority_data[c]["allowed_changes"] for c in cards]
    p_icu = [admission_data[c]["p_icu"] for c in cards if c in admission_data]
    p_ward = [admission_data[c]["p_ward"] for c in cards if c in admission_data]
    shared_cards = [c for c in cards if c in admission_data]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Operate-By Day vs. Allowed Changes",
            "P(ICU) vs. P(Ward) per Operation Card",
            "Operate-By Day Distribution",
            "ICU & Ward Admission Probabilities",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    # Scatter: operate_by vs allowed_changes
    hover_prio = [
        f"<b>{c}</b><br>Operate by: day {ob}<br>Allowed changes: {ac}"
        for c, ob, ac in zip(cards, operate_by, allowed_changes)
    ]
    fig.add_trace(
        go.Scatter(
            x=operate_by,
            y=allowed_changes,
            mode="markers",
            marker=dict(
                color=allowed_changes,
                colorscale="RdYlGn",
                size=7,
                opacity=0.75,
                showscale=True,
                colorbar=dict(title="Changes", x=0.44, len=0.45, y=0.78),
            ),
            text=hover_prio,
            hovertemplate="%{text}<extra></extra>",
            name="Op cards",
        ),
        row=1,
        col=1,
    )

    # Scatter: p_icu vs p_ward
    hover_adm = [
        f"<b>{c}</b><br>P(ICU): {pi:.3f}<br>P(Ward): {pw:.3f}"
        for c, pi, pw in zip(shared_cards, p_icu, p_ward)
    ]
    fig.add_trace(
        go.Scatter(
            x=p_icu,
            y=p_ward,
            mode="markers",
            marker=dict(
                color="#a78bfa",
                size=6,
                opacity=0.7,
            ),
            text=hover_adm,
            hovertemplate="%{text}<extra></extra>",
            name="Admission",
        ),
        row=1,
        col=2,
    )

    # Histogram: operate_by distribution
    fig.add_trace(
        go.Histogram(
            x=operate_by,
            nbinsx=25,
            marker_color="#34d399",
            hovertemplate="By day %{x}<br>Count: %{y}<extra></extra>",
            name="Operate-By",
        ),
        row=2,
        col=1,
    )

    # Bar: mean p_icu and p_ward with individual points
    fig.add_trace(
        go.Histogram(
            x=p_icu,
            nbinsx=20,
            name="P(ICU)",
            marker_color="#ef4444",
            opacity=0.75,
            hovertemplate="P(ICU) = %{x:.3f}<br>Count: %{y}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=p_ward,
            nbinsx=20,
            name="P(Ward)",
            marker_color="#3b82f6",
            opacity=0.75,
            hovertemplate="P(Ward) = %{x:.3f}<br>Count: %{y}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        barmode="overlay",
        showlegend=True,
        legend=dict(x=1.02, y=0.1),
        title_text="Priority & Admission Data",
        title_font_size=18,
    )
    fig.update_xaxes(title_text="Operate-By Day", row=1, col=1)
    fig.update_yaxes(title_text="Allowed Changes", row=1, col=1)
    fig.update_xaxes(title_text="P(ICU)", row=1, col=2)
    fig.update_yaxes(title_text="P(Ward)", row=1, col=2)
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Probability", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def visualize(output_path: str | Path = "surgery_data_dashboard.html") -> Path:
    """Generate all data, build figures, write a self-contained HTML dashboard."""
    print("Generating data…")
    data = generate_all_data(
        n_rooms=5,
        n_surgeons=10,
        or_capacity=480.0,
        n_operation_cards=100,
        waiting_list_size=50,
        seed=42,
        frequency_params=pm.FrequencyParams(),
        duration_params=pm.DurationParams(),
        schedule_params=pm.ScheduleParams(),
        priority_params=pm.PriorityParams(),
        admission_params=pm.AdmissionParams(),
    )
    (
        frequency_data,
        duration_data,
        schedule,
        priority_data,
        admission_data,
        waiting_list,
    ) = data

    print("Building figures…")
    sections = [
        ("🏥 Waiting List Summary", plot_waiting_list(waiting_list)),
        ("📊 Case Mix & Surgeon–Card Frequencies", plot_case_mix(frequency_data)),
        ("📅 Weekly Schedule Distribution", plot_schedule(schedule, n_surgeons=10)),
        ("⏱ Duration Parameters", plot_duration_data(duration_data)),
        (
            "⚖️ Priority & Admission",
            plot_priority_admission(priority_data, admission_data),
        ),
    ]

    # Build HTML
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    body = "\n".join(_html_section(title, fig) for title, fig in sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Surgery Generator – Data Dashboard</title>
  {plotly_cdn}
  <style>
    body {{
      background: #111827;
      color: #f9fafb;
      font-family: sans-serif;
      margin: 0;
      padding: 24px 32px 64px;
    }}
    h1 {{
      font-size: 1.8rem;
      border-bottom: 2px solid #374151;
      padding-bottom: 8px;
      margin-bottom: 0;
    }}
    p.subtitle {{
      color: #9ca3af;
      margin-top: 4px;
    }}
  </style>
</head>
<body>
  <h1>Surgery Generator – Data Dashboard</h1>
  <p class="subtitle">
    Seed 42 · 5 rooms · 10 surgeons · 100 operation cards · 50-surgery waiting list
  </p>
  {body}
</body>
</html>"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard saved → {out.resolve()}")
    webbrowser.open(out.resolve().as_uri())
    return out


if __name__ == "__main__":
    visualize()
