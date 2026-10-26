"""Light/dark theme: selector, Streamlit theme options, and WCAG contrast of both palettes."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit import config as st_config  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from docintel.config.settings import PROJECT_ROOT  # noqa: E402
from docintel.ui import theme  # noqa: E402
from docintel.ui.state import PAGE_FILES  # noqa: E402
from tests.conftest import make_factory  # noqa: E402

KEYS = sorted({k for p in theme.PALETTES.values() for k in p})


@pytest.fixture(autouse=True)
def restore_theme_config():
    saved = {k: st_config.get_option(f"theme.{k}") for k in KEYS}
    yield
    for key, value in saved.items():
        st_config.set_option(f"theme.{key}", value)


def _app(settings) -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    at.session_state["settings"] = settings
    at.session_state["engine_factory"] = make_factory(settings)
    return at


def test_light_is_default(settings):
    at = _app(settings).run()
    assert not at.exception
    assert st_config.get_option("theme.base") == "light"
    assert at.sidebar.segmented_control[0].value == "light"


def test_selector_switches_theme_both_ways(settings):
    at = _app(settings).run()
    at.sidebar.segmented_control[0].set_value("dark").run()
    assert not at.exception, [e.value for e in at.exception]
    assert st_config.get_option("theme.base") == "dark"
    assert st_config.get_option("theme.backgroundColor") == theme.PALETTES["dark"]["backgroundColor"]
    assert st_config.get_option("theme.chartCategoricalColors") == theme.PALETTES["dark"]["chartCategoricalColors"]
    assert st_config.get_option("theme.sidebar.backgroundColor") == theme.PALETTES["dark"]["sidebar.backgroundColor"]
    assert at.session_state[theme.SESSION_KEY] == "dark"
    at.sidebar.segmented_control[0].set_value("light").run()
    assert st_config.get_option("theme.base") == "light"


def test_configured_default_dark(settings):
    at = _app(settings.model_copy(update={"ui_theme": "dark"})).run()
    assert not at.exception
    assert st_config.get_option("theme.base") == "dark"
    assert at.sidebar.segmented_control[0].value == "dark"


@pytest.mark.parametrize("page", list(PAGE_FILES))
def test_every_page_renders_in_dark_mode(settings, page):
    at = _app(settings.model_copy(update={"ui_theme": "dark"})).run()
    at.switch_page(PAGE_FILES[page]).run()
    assert not at.exception, [e.value for e in at.exception]


def _contrast(a: str, b: str) -> float:
    def lum(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_palette_contrast_meets_wcag_aa(mode):
    p, t = theme.PALETTES[mode], theme.TOKENS[mode]
    bg = p["backgroundColor"]
    text_pairs = [(p["textColor"], bg), (p["textColor"], p["secondaryBackgroundColor"]),
                  (p["sidebar.textColor"], p["sidebar.backgroundColor"]), (p["linkColor"], bg),
                  ("#ffffff", p["primaryColor"]), (t["muted"], bg), (t["ink_2"], bg), (t["low_fg"], t["low_bg"]),
                  (t["ent_fg"], t["ent_bg"]), (t["bad_fg"], t["bad_bg"])]
    text_pairs += [(p[c], bg) for c in ("greenColor", "redColor", "orangeColor", "blueColor", "grayColor", "violetColor", "yellowColor")]
    for fg, back in text_pairs:
        assert _contrast(fg, back) >= 4.5, (mode, fg, back)
    assert _contrast(p["primaryColor"], bg) >= 3.0  # active/selected UI elements
