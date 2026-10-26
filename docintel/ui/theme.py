"""Light/dark theme for the whole app, applied through Streamlit's native theme options.

All colours (text, borders, status badges, charts, dataframes, sidebar) are Streamlit
``theme.*`` options, so native widgets follow the theme without custom CSS.

Switching at runtime: Streamlit has no public API for a widget to change the theme. It
does re-read the ``theme.*`` options at the start of every script run and sends them to
the browser, so :func:`apply` updates those options and the caller reruns once. The
options are process-wide: in this single-user local app every open tab follows the latest
choice. Contrast of every text/status colour against its background is checked in
``tests/unit/test_ui_theme.py`` (WCAG AA: 4.5:1 for text, 3:1 for UI and status colours).
"""

from __future__ import annotations

import logging
from typing import Literal

import streamlit as st

logger = logging.getLogger(__name__)

Mode = Literal["light", "dark"]
# The mode lives in a plain session key; the selector widget has its own key and copies its
# value here in on_change. Widget state is discarded when a run is cut short by st.rerun()
# before the widget is drawn, which would otherwise silently undo the user's choice.
SESSION_KEY = "ui_theme_mode"
SELECTOR_KEY = "ui_theme_selector"
LABELS: dict[Mode, str] = {"light": ":material/light_mode: Light", "dark": ":material/dark_mode: Dark"}

# Chart palette: validated reference-palette slots (blue, aqua, critical red) per surface.
PALETTES: dict[Mode, dict[str, object]] = {
    "light": {
        "base": "light",
        "primaryColor": "#1f5bd6",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f2f4f8",
        "textColor": "#101828",
        "linkColor": "#1f5bd6",
        "borderColor": "#d0d5dd",
        "codeBackgroundColor": "#f2f4f8",
        "dataframeHeaderBackgroundColor": "#f2f4f8",
        "dataframeBorderColor": "#e4e7ec",
        "greenColor": "#067647",
        "redColor": "#b42318",
        "orangeColor": "#b54708",
        "blueColor": "#1f5bd6",
        "grayColor": "#475467",
        "violetColor": "#6941c6",
        "yellowColor": "#a15c07",
        "chartCategoricalColors": ["#2a78d6", "#1baf7a", "#d03b3b", "#eda100", "#e87ba4", "#008300", "#6250d6", "#eb6834"],
        "baseRadius": "8px",
        "buttonRadius": "8px",
        "headingFontSizes": ["1.9rem", "1.45rem", "1.2rem", "1.05rem", "0.95rem", "0.85rem"],
        "headingFontWeights": [700, 650, 600, 600, 600, 600],
        "showWidgetBorder": True,
        "showSidebarBorder": True,
        "sidebar.backgroundColor": "#f7f8fa",
        "sidebar.secondaryBackgroundColor": "#eceff4",
        "sidebar.borderColor": "#d0d5dd",
        "sidebar.textColor": "#101828",
    },
    "dark": {
        "base": "dark",
        "primaryColor": "#2f67d8",
        "backgroundColor": "#0f1520",
        "secondaryBackgroundColor": "#1a2230",
        "textColor": "#e6eaf2",
        "linkColor": "#8ab4ff",
        "borderColor": "#344054",
        "codeBackgroundColor": "#161d29",
        "dataframeHeaderBackgroundColor": "#1a2230",
        "dataframeBorderColor": "#2a3343",
        "greenColor": "#47cd89",
        "redColor": "#f97066",
        "orangeColor": "#fdb022",
        "blueColor": "#8ab4ff",
        "grayColor": "#98a2b3",
        "violetColor": "#b692f6",
        "yellowColor": "#fec84b",
        "chartCategoricalColors": ["#3987e5", "#199e70", "#d03b3b", "#c98500", "#d55181", "#008300", "#9085e9", "#d95926"],
        "baseRadius": "8px",
        "buttonRadius": "8px",
        "headingFontSizes": ["1.9rem", "1.45rem", "1.2rem", "1.05rem", "0.95rem", "0.85rem"],
        "headingFontWeights": [700, 650, 600, 600, 600, 600],
        "showWidgetBorder": True,
        "showSidebarBorder": True,
        "sidebar.backgroundColor": "#0b1019",
        "sidebar.secondaryBackgroundColor": "#161d29",
        "sidebar.borderColor": "#2a3343",
        "sidebar.textColor": "#e6eaf2",
    },
}

# Colours for DocIntel's own small pieces of HTML (low-confidence highlighting) and charts.
TOKENS: dict[Mode, dict[str, str]] = {
    "light": {"surface": "#ffffff", "ink": "#101828", "ink_2": "#475467", "muted": "#667085", "grid": "#e4e7ec",
              "axis": "#d0d5dd", "series_1": "#2a78d6", "series_3": "#1baf7a", "critical": "#d03b3b",
              "low_bg": "#fef0c7", "low_fg": "#7a2e0e", "low_bd": "#dc6803", "raised": "#f9fafb", "line": "#e4e7ec",
              "ent_bg": "#e0ecfb", "ent_fg": "#0b3b7a", "ent_bd": "#2a78d6",
              "bad_bg": "#fde2e1", "bad_fg": "#7a1414", "bad_bd": "#d03b3b"},
    "dark": {"surface": "#0f1520", "ink": "#e6eaf2", "ink_2": "#c0c7d4", "muted": "#98a2b3", "grid": "#2a3343",
             "axis": "#344054", "series_1": "#3987e5", "series_3": "#199e70", "critical": "#d03b3b",
             "low_bg": "#4e3308", "low_fg": "#fedf89", "low_bd": "#f79009", "raised": "#141b27", "line": "#2a3343",
             "ent_bg": "#12305a", "ent_fg": "#cfe1fb", "ent_bd": "#3987e5",
             "bad_bg": "#4d1515", "bad_fg": "#fcd0cd", "bad_bd": "#e5534b"},
}


def mode() -> Mode:
    return "dark" if st.session_state.get(SESSION_KEY) == "dark" else "light"


def tokens(current: Mode | None = None) -> dict[str, str]:
    return TOKENS[current or mode()]


def init(default: Mode) -> None:
    """Seed the session's mode from the configured default (DOCINTEL_UI_THEME)."""
    st.session_state.setdefault(SESSION_KEY, default)


def _on_select() -> None:
    choice = st.session_state.get(SELECTOR_KEY)
    if choice in ("light", "dark"):
        st.session_state[SESSION_KEY] = choice


def selector() -> None:
    """Sidebar theme selector (state is kept in SESSION_KEY, see module note)."""
    st.session_state[SELECTOR_KEY] = mode()
    st.segmented_control("Theme", ["light", "dark"], format_func=lambda m: LABELS[m], key=SELECTOR_KEY,
                         on_change=_on_select, required=True, width="stretch")


def apply() -> bool:
    """Push the session's palette into Streamlit's theme options.

    Returns True if an option changed. The browser only receives the new theme at the start
    of the next script run, so the caller should then call ``st.rerun()``.
    """
    changed = False
    try:
        from streamlit import config as st_config

        for name, value in PALETTES[mode()].items():
            key = f"theme.{name}"
            if st_config.get_option(key) != value:
                st_config.set_option(key, value)
                changed = True
    except (ImportError, AttributeError, KeyError) as exc:  # internal API moved in a future Streamlit
        logger.warning("Could not switch Streamlit theme", extra={"reason": str(exc)})
    return changed
