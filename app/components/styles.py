"""
Dashboard Theme & Component Library
===================================
A single source of truth for the dashboard's visual language.

The design target is an *instrument panel*, not a marketing page: flat surfaces,
hairline rules, tabular figures, and a small reserved status palette. Colour is
never the only carrier of meaning — every status pill ships a glyph and a label.

Contents
--------
``TOKENS``            Design tokens, also consumed by the Plotly theme.
``get_custom_css()``  The stylesheet injected once by ``app/main.py``.
``render_*()``        HTML fragment builders for the shared components.
``plotly_layout()``   Chart layout defaults so figures match the page chrome.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ============================================================================
# Design tokens
# ----------------------------------------------------------------------------
# Surfaces and ink are tuned for the dark industrial plane. The categorical and
# status ramps are validated for colour-vision deficiency against the card
# surface (#151a21): worst adjacent CVD dE 8.4, worst normal-vision dE 19.3, and
# every slot clears 3:1 contrast. Do not substitute colours here without
# re-validating — several pages read these values directly.
# ============================================================================

TOKENS: Dict[str, str] = {
    # Planes
    "page": "#0e1217",
    "surface": "#151a21",
    "surface_raised": "#1b212a",
    "sidebar": "#0b0f14",
    # Lines
    "border": "#232b36",
    "border_strong": "#313b49",
    "grid": "#1e252f",
    # Ink
    "ink": "#e8eaed",
    "ink_secondary": "#a8b2c0",
    "ink_muted": "#6b7688",
    # Status — reserved, never reused as a series colour
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    # Accent (interactive affordances only)
    "accent": "#3987e5",
}

# Categorical series order. Assign in this order and never cycle: a 9th series
# folds into "Other" or becomes a small multiple. Charts that compare every pair
# at once (scatter, bubble) cap at the first three slots.
SERIES: list[str] = [
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
]

# Single-hue ramp for magnitude (light -> dark), used by heatmaps and choropleths.
SEQUENTIAL: list[str] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]

#: Maps a risk category label to its status token key.
RISK_STATUS: Dict[str, str] = {
    "LOW RISK": "good",
    "MODERATE RISK": "warning",
    "HIGH RISK": "serious",
    "CRITICAL RISK": "critical",
}


def _t(key: str) -> str:
    """Look up a design token, raising loudly on a typo."""
    return TOKENS[key]


# ============================================================================
# Stylesheet
# ============================================================================

def get_custom_css() -> str:
    """Return the complete stylesheet for the dashboard.

    Injected once from ``app/main.py`` immediately after ``set_page_config``.
    """
    t = TOKENS
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --page: {t["page"]};
        --surface: {t["surface"]};
        --surface-raised: {t["surface_raised"]};
        --sidebar: {t["sidebar"]};

        --border: {t["border"]};
        --border-strong: {t["border_strong"]};
        --grid: {t["grid"]};

        --ink: {t["ink"]};
        --ink-secondary: {t["ink_secondary"]};
        --ink-muted: {t["ink_muted"]};

        --good: {t["good"]};
        --warning: {t["warning"]};
        --serious: {t["serious"]};
        --critical: {t["critical"]};
        --accent: {t["accent"]};

        --radius: 4px;
        --radius-lg: 6px;

        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace;
    }}

    /* ---------------------------------------------------------------- base */

    html, body, [class*="css"] {{
        font-family: var(--font-sans) !important;
        color: var(--ink) !important;
    }}

    .stApp {{ background: var(--page) !important; }}

    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1480px !important;
    }}

    code, pre, kbd {{ font-family: var(--font-mono) !important; }}

    /* Numeric readouts align in columns; prose keeps proportional figures. */
    .metric-value, .data-figure, [data-testid="stMetricValue"],
    .stDataFrame td, .stDataFrame th {{
        font-variant-numeric: tabular-nums;
    }}

    h1, h2, h3, h4, h5 {{
        color: var(--ink) !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }}
    h1 {{ font-size: 1.55rem !important; }}
    h2 {{ font-size: 1.2rem !important; }}
    h3 {{ font-size: 1.0rem !important; }}
    h4 {{ font-size: 0.9rem !important; }}

    p, li, label, .stMarkdown {{ color: var(--ink-secondary); }}

    hr {{ border-color: var(--border) !important; }}

    /* ------------------------------------------------------------- sidebar */

    section[data-testid="stSidebar"] {{
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }}

    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem !important; }}

    /* Streamlit's own multipage nav duplicates our router — hide it. */
    div[data-testid="stSidebarNav"], nav[data-testid="stSidebarNav"] {{ display: none !important; }}

    section[data-testid="stSidebar"] .stRadio > div {{ gap: 1px !important; }}

    section[data-testid="stSidebar"] .stRadio > div > label {{
        padding: 8px 12px !important;
        border-radius: var(--radius) !important;
        cursor: pointer !important;
        background: transparent !important;
        border-left: 2px solid transparent !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        color: var(--ink-secondary) !important;
        transition: background-color 0.12s ease, color 0.12s ease !important;
    }}

    section[data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: rgba(255, 255, 255, 0.035) !important;
        color: var(--ink) !important;
    }}

    section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
        background: rgba(57, 135, 229, 0.1) !important;
        border-left-color: var(--accent) !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
    }}

    /* The radio dot is redundant next to the selected row's accent rule. */
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}

    /* ------------------------------------------------------- page masthead */

    .masthead {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 24px;
        padding: 0 0 14px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--border);
    }}

    .masthead h1 {{
        margin: 0 !important;
        font-size: 1.4rem !important;
        line-height: 1.25;
    }}

    .masthead .masthead-sub {{
        color: var(--ink-secondary);
        font-size: 0.83rem;
        margin-top: 5px;
        max-width: 72ch;
    }}

    .masthead .masthead-meta {{
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--ink-muted);
        white-space: nowrap;
        text-align: right;
        padding-top: 4px;
    }}

    /* ----------------------------------------------------------- stat tile */

    .stat-tile {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 14px 16px;
        height: 100%;
    }}

    .stat-tile .stat-label {{
        font-size: 0.67rem;
        font-weight: 600;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .stat-tile .stat-value {{
        font-size: 1.6rem;
        font-weight: 600;
        line-height: 1.1;
        color: var(--ink);
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }}

    .stat-tile .stat-note {{
        font-size: 0.73rem;
        color: var(--ink-secondary);
        margin-top: 7px;
        line-height: 1.4;
    }}

    /* A 3px rail carries the status; the glyph + label in the note carries the
       meaning, so the colour is never doing the work alone. */
    .stat-tile.is-good     {{ box-shadow: inset 3px 0 0 var(--good); }}
    .stat-tile.is-warning  {{ box-shadow: inset 3px 0 0 var(--warning); }}
    .stat-tile.is-serious  {{ box-shadow: inset 3px 0 0 var(--serious); }}
    .stat-tile.is-critical {{ box-shadow: inset 3px 0 0 var(--critical); }}
    .stat-tile.is-neutral  {{ box-shadow: inset 3px 0 0 var(--border-strong); }}
    .stat-tile.is-accent   {{ box-shadow: inset 3px 0 0 var(--accent); }}

    .stat-tile.is-good     .stat-value {{ color: var(--good); }}
    .stat-tile.is-warning  .stat-value {{ color: var(--warning); }}
    .stat-tile.is-serious  .stat-value {{ color: var(--serious); }}
    .stat-tile.is-critical .stat-value {{ color: var(--critical); }}

    /* --------------------------------------------------------------- panel */

    .panel {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 16px 18px 18px 18px;
        margin-bottom: 16px;
    }}

    .panel-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 10px;
        margin-bottom: 14px;
        border-bottom: 1px solid var(--border);
    }}

    .panel-head .panel-title {{
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--ink);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .panel-head .panel-note {{
        font-size: 0.73rem;
        color: var(--ink-muted);
    }}

    /* --------------------------------------------------------------- pills */

    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        white-space: nowrap;
        font-family: var(--font-sans);
    }}

    .pill .pill-glyph {{ font-size: 0.8em; line-height: 1; }}

    .pill-good     {{ background: rgba(12, 163, 12, 0.14);  color: var(--good);     border: 1px solid rgba(12, 163, 12, 0.3); }}
    .pill-warning  {{ background: rgba(250, 178, 25, 0.13); color: var(--warning);  border: 1px solid rgba(250, 178, 25, 0.3); }}
    .pill-serious  {{ background: rgba(236, 131, 90, 0.13); color: var(--serious);  border: 1px solid rgba(236, 131, 90, 0.3); }}
    .pill-critical {{ background: rgba(208, 59, 59, 0.15);  color: var(--critical); border: 1px solid rgba(208, 59, 59, 0.35); }}
    .pill-neutral  {{ background: rgba(168, 178, 192, 0.1); color: var(--ink-secondary); border: 1px solid var(--border-strong); }}
    .pill-accent   {{ background: rgba(57, 135, 229, 0.13); color: var(--accent);   border: 1px solid rgba(57, 135, 229, 0.3); }}

    /* -------------------------------------------------------------- notice */

    .notice {{
        border: 1px solid var(--border);
        border-left: 3px solid var(--border-strong);
        border-radius: var(--radius);
        padding: 12px 16px;
        margin: 0 0 16px 0;
        background: var(--surface);
    }}

    .notice .notice-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 3px;
    }}

    .notice .notice-body {{
        font-size: 0.8rem;
        color: var(--ink-secondary);
        line-height: 1.5;
    }}

    .notice-good     {{ border-left-color: var(--good); }}
    .notice-warning  {{ border-left-color: var(--warning); }}
    .notice-serious  {{ border-left-color: var(--serious); }}
    .notice-critical {{ border-left-color: var(--critical); }}
    .notice-accent   {{ border-left-color: var(--accent); }}

    .notice-good     .notice-title {{ color: var(--good); }}
    .notice-warning  .notice-title {{ color: var(--warning); }}
    .notice-serious  .notice-title {{ color: var(--serious); }}
    .notice-critical .notice-title {{ color: var(--critical); }}

    /* --------------------------------------------------------- action card */

    .action {{
        border: 1px solid var(--border);
        border-left: 3px solid var(--border-strong);
        border-radius: var(--radius);
        padding: 12px 16px;
        margin-bottom: 10px;
        background: var(--surface);
    }}

    .action-critical {{ border-left-color: var(--critical); }}
    .action-high     {{ border-left-color: var(--serious); }}
    .action-medium   {{ border-left-color: var(--warning); }}
    .action-low      {{ border-left-color: var(--good); }}

    .action .action-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 6px;
    }}

    .action .action-title {{ font-size: 0.87rem; font-weight: 600; color: var(--ink); }}
    .action .action-trigger {{ font-size: 0.74rem; color: var(--ink-muted); margin-bottom: 6px; }}
    .action ul {{ margin: 0 0 6px 0; padding-left: 18px; }}
    .action li {{ font-size: 0.8rem; color: var(--ink-secondary); margin-bottom: 2px; }}
    .action .action-context {{
        font-size: 0.73rem;
        color: var(--ink-muted);
        border-top: 1px solid var(--border);
        padding-top: 6px;
        margin-top: 8px;
    }}

    /* ------------------------------------------------------- key/value rows */

    .kv {{ display: flex; justify-content: space-between; gap: 16px; padding: 5px 0; font-size: 0.79rem; }}
    .kv + .kv {{ border-top: 1px solid var(--border); }}
    .kv .kv-key {{ color: var(--ink-muted); }}
    .kv .kv-val {{ color: var(--ink); font-variant-numeric: tabular-nums; font-weight: 500; }}

    /* --------------------------------------------------- streamlit widgets */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 34px;
        padding: 0 14px;
        background: transparent !important;
        border-radius: var(--radius) var(--radius) 0 0;
        color: var(--ink-muted) !important;
        font-size: 0.81rem !important;
        font-weight: 500 !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--ink) !important;
        border-bottom: 2px solid var(--accent) !important;
        font-weight: 600 !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}

    .stButton > button {{
        border-radius: var(--radius) !important;
        border: 1px solid var(--border-strong) !important;
        background: var(--surface-raised) !important;
        color: var(--ink) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.42rem 0.9rem !important;
        transition: border-color 0.12s ease, background-color 0.12s ease !important;
    }}

    .stButton > button:hover {{
        border-color: var(--accent) !important;
        background: var(--surface-raised) !important;
        color: var(--ink) !important;
    }}

    .stButton > button[kind="primary"] {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    .stButton > button[kind="primary"]:hover {{ background: #2f76ce !important; border-color: #2f76ce !important; }}

    .stDownloadButton > button {{
        border-radius: var(--radius) !important;
        border: 1px solid var(--border-strong) !important;
        background: var(--surface-raised) !important;
        color: var(--ink) !important;
        font-size: 0.8rem !important;
    }}

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-baseweb="select"] > div {{
        background: var(--surface-raised) !important;
        border-color: var(--border-strong) !important;
        color: var(--ink) !important;
        font-size: 0.83rem !important;
    }}

    div[data-testid="stNumberInput"] input {{ font-family: var(--font-mono) !important; }}

    div[data-testid="stWidgetLabel"] label p {{
        font-size: 0.77rem !important;
        color: var(--ink-secondary) !important;
        font-weight: 500 !important;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        background: var(--surface) !important;
    }}

    div[data-testid="stExpander"] summary p {{ font-size: 0.8rem !important; font-weight: 500 !important; }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }}

    /* Streamlit's stock alerts, toned down to match the panels. */
    div[data-testid="stAlert"] {{
        border-radius: var(--radius) !important;
        border-width: 1px !important;
        font-size: 0.82rem !important;
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--accent) !important; }}

    /* Streamlit chrome we replace with our own. */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}

    /* -------------------------------------------------------------- footer */

    .page-footer {{
        margin-top: 34px;
        padding-top: 14px;
        border-top: 1px solid var(--border);
        color: var(--ink-muted);
        font-size: 0.72rem;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
    }}

    .disclaimer {{
        font-size: 0.73rem;
        color: var(--ink-muted);
        border-top: 1px solid var(--border);
        padding-top: 10px;
        margin-top: 12px;
        line-height: 1.5;
    }}

    /* Keep the layout usable on a narrow laptop / split screen. */
    @media (max-width: 900px) {{
        .masthead {{ flex-direction: column; gap: 8px; }}
        .masthead .masthead-meta {{ text-align: left; }}
        .stat-tile .stat-value {{ font-size: 1.35rem; }}
    }}
    </style>
    """


# ============================================================================
# Component builders
# ============================================================================

#: Glyphs pair with every status colour so meaning survives a greyscale print
#: or a colour-vision deficiency.
_STATUS_GLYPH = {
    "good": "●",      # filled circle
    "warning": "▲",   # triangle
    "serious": "◆",   # diamond
    "critical": "■",  # square
    "neutral": "○",   # hollow circle
    "accent": "●",
}

_VALID_STATUS = set(_STATUS_GLYPH)


def _status(value: str) -> str:
    """Normalise an arbitrary status string to a known token key."""
    key = (value or "neutral").strip().lower()
    return key if key in _VALID_STATUS else "neutral"


def render_stat_tile(
    label: str,
    value: str,
    note: str = "",
    status: str = "neutral",
) -> str:
    """Build a single KPI tile.

    Args:
        label: Short uppercase caption, e.g. ``"Monitored assets"``.
        value: The figure itself. Rendered with tabular numerals.
        note: Supporting line beneath the value. May contain inline HTML.
        status: One of ``good``, ``warning``, ``serious``, ``critical``,
            ``accent`` or ``neutral``. Drives the left rail and value colour.
    """
    note_html = f'<div class="stat-note">{note}</div>' if note else ""
    return (
        f'<div class="stat-tile is-{_status(status)}">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f"{note_html}"
        f"</div>"
    )


def render_pill(text: str, status: str = "neutral") -> str:
    """Build a status pill. The glyph makes the state readable without colour."""
    key = _status(status)
    return (
        f'<span class="pill pill-{key}">'
        f'<span class="pill-glyph">{_STATUS_GLYPH[key]}</span>{text}'
        f"</span>"
    )


def render_risk_pill(category: str) -> str:
    """Build a pill for a risk category label such as ``"CRITICAL RISK"``."""
    return render_pill(category, RISK_STATUS.get(str(category).upper(), "neutral"))


def render_masthead(title: str, subtitle: str = "", meta: str = "") -> str:
    """Build the page masthead — title, one-line purpose, right-aligned meta."""
    sub_html = f'<div class="masthead-sub">{subtitle}</div>' if subtitle else ""
    meta_html = f'<div class="masthead-meta">{meta}</div>' if meta else ""
    return (
        f'<div class="masthead"><div><h1>{title}</h1>{sub_html}</div>{meta_html}</div>'
    )


def render_panel_head(title: str, note: str = "") -> str:
    """Build a section header rule for use inside a panel."""
    note_html = f'<div class="panel-note">{note}</div>' if note else ""
    return f'<div class="panel-head"><div class="panel-title">{title}</div>{note_html}</div>'


def render_notice(title: str, body: str, status: str = "accent") -> str:
    """Build a bordered notice block (alerts, findings, callouts)."""
    key = _status(status)
    return (
        f'<div class="notice notice-{key}">'
        f'<div class="notice-title">{_STATUS_GLYPH[key]} {title}</div>'
        f'<div class="notice-body">{body}</div>'
        f"</div>"
    )


def render_kv_rows(rows: Dict[str, Any]) -> str:
    """Build a compact key/value list — a readable alternative to a 2-row table."""
    return "".join(
        f'<div class="kv"><span class="kv-key">{k}</span>'
        f'<span class="kv-val">{v}</span></div>'
        for k, v in rows.items()
    )


def render_validation_item(text: str, is_valid: bool = True) -> str:
    """Build one line of a data-validation checklist."""
    key = "good" if is_valid else "warning"
    return (
        f'<div style="margin: 3px 0; font-size: 0.8rem; color: var(--ink-secondary);">'
        f'<span style="color: var(--{key});">{_STATUS_GLYPH[key]}</span> {text}</div>'
    )


def render_footer(left: str, right: str = "") -> str:
    """Build the page footer strip."""
    return f'<div class="page-footer"><div>{left}</div><div>{right}</div></div>'


# ============================================================================
# Plotly theme
# ============================================================================

def plotly_layout(
    height: int = 300,
    show_legend: bool = False,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    margin: Optional[Dict[str, int]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return Plotly layout defaults matching the page chrome.

    Every figure in the app passes through here so axes, grid weight, fonts and
    hover styling stay consistent — and so a theme change is a one-file edit.

    Args:
        height: Figure height in pixels.
        show_legend: Legend visibility. A single-series chart names itself in
            the panel header and needs no legend box.
        x_title: X axis title. Omit when the tick labels are self-evident.
        y_title: Y axis title.
        margin: Override the default margins.
        **overrides: Any further Plotly layout keys, merged last.

    Returns:
        A kwargs dict for ``fig.update_layout(**plotly_layout(...))``.
    """
    t = TOKENS
    axis = dict(
        gridcolor=t["grid"],
        zerolinecolor=t["border_strong"],
        linecolor=t["border_strong"],
        tickfont=dict(size=11, color=t["ink_muted"]),
        title_font=dict(size=11, color=t["ink_muted"]),
        showline=True,
    )

    layout: Dict[str, Any] = dict(
        height=height,
        margin=margin or dict(t=16, b=40, l=52, r=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=t["ink_secondary"]),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11, color=t["ink_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=t["surface_raised"],
            bordercolor=t["border_strong"],
            font=dict(family="Inter, sans-serif", size=12, color=t["ink"]),
        ),
        xaxis={**axis, "title": x_title},
        yaxis={**axis, "title": y_title},
        colorway=SERIES,
    )
    layout.update(overrides)
    return layout


def style_figure(fig, **kwargs: Any):
    """Apply :func:`plotly_layout` to a figure in place and return it."""
    fig.update_layout(**plotly_layout(**kwargs))
    return fig


__all__ = [
    "TOKENS",
    "SERIES",
    "SEQUENTIAL",
    "RISK_STATUS",
    "get_custom_css",
    "render_stat_tile",
    "render_pill",
    "render_risk_pill",
    "render_masthead",
    "render_panel_head",
    "render_notice",
    "render_kv_rows",
    "render_validation_item",
    "render_footer",
    "plotly_layout",
    "style_figure",
]
