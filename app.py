"""DocIntel Streamlit application.

Run with either of:
    python app.py                      (starts the server and opens the browser)
    python -m streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if __name__ == "__main__":
    # Streamlit reads .streamlit/config.toml from the working directory when it is first
    # imported, so switch to the project folder *before* importing it.
    os.chdir(PROJECT_DIR)

try:
    import streamlit as st
except ImportError:
    sys.exit(
        "DocIntel needs its Python libraries, which are not installed for this Python:\n"
        f"    {sys.executable}\n"
        "Install them with:\n"
        f'    "{sys.executable}" -m pip install -r requirements.txt\n'
        "or run the app with the virtual environment's python.exe (see README)."
    )


def _open_browser_when_ready() -> None:
    """Open the app in the default browser once the server answers its health check."""
    import time
    import urllib.request
    import webbrowser

    from streamlit import config as st_config

    for _ in range(120):  # up to ~60 s
        time.sleep(0.5)
        port = st_config.get_option("server.port")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1):  # noqa: S310 - fixed localhost URL
                webbrowser.open(f"http://localhost:{port}")
                return
        except OSError:
            continue


def _launch_server() -> None:
    """`python app.py` (or an IDE's Run button): start Streamlit for this file.

    Streamlit runs headless, which skips its interactive first-run prompts (newsletter
    e-mail question, "streamlit skills" promotion); the browser is opened by this launcher
    instead. Privacy/security settings are passed explicitly so they hold even if
    .streamlit/config.toml is missing: localhost only, no usage statistics.
    """
    import threading

    from streamlit.web import cli as stcli

    user_args = sys.argv[1:]
    if "--no-browser" in user_args:
        user_args.remove("--no-browser")
    else:
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    sys.argv = [
        "streamlit", "run", str(PROJECT_DIR / "app.py"),
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true",
        *user_args,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    from streamlit import runtime

    if not runtime.exists():  # plain `python app.py`; under `streamlit run` the runtime already exists
        _launch_server()

st.set_page_config(page_title="DocIntel", page_icon=":material/document_scanner:", layout="wide")

from docintel import __version__  # noqa: E402
from docintel.config.logging_config import configure_logging  # noqa: E402
from docintel.ui import state, theme  # noqa: E402

ASSETS = PROJECT_DIR / "docintel" / "ui" / "assets"


def _page(key: str, title: str, icon: str, **kwargs) -> st.Page:
    return st.Page(state.PAGE_FILES[key], title=title, icon=icon, **kwargs)


def main() -> None:
    settings = state.settings()
    configure_logging(settings)
    theme.init(settings.ui_theme)
    if theme.apply():
        # Streamlit sends theme options at the start of a run, so the new theme reaches the
        # browser on the next run. Rerun once now instead of waiting for the next interaction.
        st.rerun()

    st.logo(str(ASSETS / f"logo_{theme.mode()}.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
    navigation = st.navigation(
        {
            "": [_page("dashboard", "Dashboard", ":material/space_dashboard:", default=True)],
            "Process": [
                _page("image_ocr", "Image OCR", ":material/document_scanner:"),
                _page("pdf_tools", "PDF tools", ":material/picture_as_pdf:"),
                _page("tables", "Table extraction", ":material/table_chart:"),
                _page("structured", "Structured extraction", ":material/data_object:"),
            ],
            "Review": [
                _page("results", "OCR results", ":material/article:"),
                _page("intelligence", "Document intelligence", ":material/manage_search:"),
                _page("enhancement", "Image enhancement", ":material/tune:"),
            ],
            "System": [
                _page("settings", "Settings", ":material/settings:"),
                _page("about", "About", ":material/info:"),
            ],
        },
        position="sidebar",
    )
    with st.sidebar:
        theme.selector()
        results = [r for r in state.results().values() if r.document is not None]
        st.caption(f"v{__version__} · {len(state.uploads())} file(s) in this session · "
                   f"{sum(len(r.document.pages) for r in results)} page(s) processed")
        st.caption(":material/lock: Runs on this computer only. OCR can make mistakes; review important text.")
    navigation.run()


main()
