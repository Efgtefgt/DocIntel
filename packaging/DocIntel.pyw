"""DocIntel desktop launcher (used by the Windows installer's Start-menu and desktop shortcuts).

Run with pythonw.exe, so no console window appears:
    pythonw.exe DocIntel.pyw            start the app (or open another window if it already runs)
    pythonw.exe DocIntel.pyw --stop     stop a running app (the uninstaller uses this)

What it does:
* starts the Streamlit server on 127.0.0.1 (this computer only) in this process;
* opens the app in its own Edge/Chrome window (--app mode), or the default browser;
* stops the server about a minute after the last DocIntel window is closed;
* uses the bundled Tesseract and Ghostscript (tools\\ next to the app) when present,
  and enables PaddleOCR once the optional add-on is installed (install_paddle.py);
* keeps user data outside the program folder, so upgrades and uninstalls never touch it:
  logs and templates in %LOCALAPPDATA%\\DocIntel, saved exports in Documents\\DocIntel.

Problems are written to %LOCALAPPDATA%\\DocIntel\\logs\\launcher.log and shown in a message box.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # app.py, docintel\, app_pages\
INSTALL_DIR = APP_DIR.parent                       # python\, app\, tools\
TOOLS_DIR = INSTALL_DIR / "tools"
STATE_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "DocIntel"
SERVER_FILE = STATE_DIR / "server.json"
LOG_HINT = r"%LOCALAPPDATA%\DocIntel\logs\launcher.log"  # shown in messages (no user name)

PREFERRED_PORT = 8531          # fixed port, so a browser "Install this site as an app" keeps working
IDLE_SHUTDOWN_SECONDS = 60     # stop this long after the last window closed
NEVER_CONNECTED_SECONDS = 600  # stop if no window ever connected (browser failed to open)
MUTEX_NAME = "Local\\DocIntelDesktopServer"

log = logging.getLogger("launcher")  # not under "docintel": the app reconfigures that logger


# --- small Windows helpers ---------------------------------------------------------------------
def message_box(text: str, *, error: bool = True) -> None:
    flags = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
    try:
        ctypes.windll.user32.MessageBoxW(None, text, "DocIntel", flags)
    except Exception:  # noqa: BLE001 - not on Windows; the log still has the message
        pass


def documents_dir() -> Path:
    """The user's Documents folder (follows OneDrive/folder redirection)."""
    try:
        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:  # CSIDL_PERSONAL
            return Path(buf.value)
    except Exception:  # noqa: BLE001
        pass
    return Path.home() / "Documents"


def take_mutex() -> bool:
    """True if this process is the only launcher starting/running a server."""
    try:
        kernel32 = ctypes.windll.kernel32
        take_mutex.handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)  # type: ignore[attr-defined]
        return kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS
    except Exception:  # noqa: BLE001
        return True


# --- server discovery --------------------------------------------------------------------------
def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as resp:  # noqa: S310
            return resp.status == 200
    except OSError:
        return False


def running_server() -> dict | None:
    try:
        info = json.loads(SERVER_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and healthy(int(info.get("port", 0))) else None


def wait_for_server(timeout: float) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = running_server()
        if info:
            return info
        time.sleep(0.5)
    return None


def free_port() -> int:
    for port in (PREFERRED_PORT, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("No free network port on 127.0.0.1.")


# --- browser window ----------------------------------------------------------------------------
def find_app_browser() -> str | None:
    """Edge (always present on Windows 10/11) or Chrome, which can open a window without tabs."""
    candidates = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates += [Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                           Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"]
    found = next((str(c) for c in candidates if c.is_file()), None)
    return found or shutil.which("msedge") or shutil.which("chrome")


def open_window(port: int) -> None:
    url = f"http://localhost:{port}/"
    browser = find_app_browser()
    if browser:
        try:
            subprocess.Popen([browser, f"--app={url}", "--window-size=1400,900"], close_fds=True)  # noqa: S603
            return
        except OSError:
            log.exception("Could not start %s", browser)
    import webbrowser

    webbrowser.open(url)


# --- server ------------------------------------------------------------------------------------
def prepare_environment() -> None:
    """Point DocIntel at the bundled tools and at per-user data folders (unless already set)."""
    tesseract = TOOLS_DIR / "tesseract" / "tesseract.exe"
    ghostscript_bin = TOOLS_DIR / "ghostscript" / "bin"
    extra_path = [str(p) for p in (tesseract.parent, ghostscript_bin) if p.is_dir()]
    if extra_path:
        os.environ["PATH"] = os.pathsep.join([*extra_path, os.environ.get("PATH", "")])
    if tesseract.is_file():
        os.environ.setdefault("DOCINTEL_TESSERACT_CMD", str(tesseract))

    templates = STATE_DIR / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    for shipped in (APP_DIR / "data" / "templates").glob("*.json"):
        if not (templates / shipped.name).exists():  # never overwrite templates the user changed
            shutil.copy2(shipped, templates / shipped.name)
    os.environ.setdefault("DOCINTEL_TEMPLATES_DIR", str(templates))
    os.environ.setdefault("DOCINTEL_LOG_DIR", str(STATE_DIR / "logs"))
    os.environ.setdefault("DOCINTEL_EXPORT_DIR", str(documents_dir() / "DocIntel"))
    # PaddleOCR is an optional add-on (install_paddle.py); switch it on once it is installed.
    import importlib.util

    if importlib.util.find_spec("paddleocr") is not None:
        os.environ.setdefault("DOCINTEL_ENABLE_PADDLE", "true")


def active_sessions() -> int | None:
    """Number of connected browser windows; None if Streamlit's internals changed."""
    try:
        from streamlit import runtime

        if not runtime.exists():
            return None
        return runtime.get_instance()._session_mgr.num_active_sessions()  # noqa: SLF001 - no public API
    except Exception:  # noqa: BLE001
        return None


def watch_and_open(port: int) -> None:
    """Open the window once the server answers, then stop the process when no window is left."""
    for _ in range(240):
        if healthy(port):
            break
        time.sleep(0.5)
    else:
        log.error("Server did not start on port %s", port)
        message_box("DocIntel could not start. Details are in\n" + LOG_HINT)
        os._exit(1)
    open_window(port)

    started, ever_connected, idle_since = time.monotonic(), False, None
    while True:
        time.sleep(5)
        count = active_sessions()
        if count is None:
            continue  # cannot tell: keep running (stops at sign-out or with --stop)
        if count > 0:
            ever_connected, idle_since = True, None
            continue
        idle_since = idle_since or time.monotonic()
        limit = IDLE_SHUTDOWN_SECONDS if ever_connected else NEVER_CONNECTED_SECONDS
        if time.monotonic() - (idle_since if ever_connected else started) >= limit:
            log.info("No DocIntel window open; stopping the server")
            SERVER_FILE.unlink(missing_ok=True)
            os._exit(0)


def run_server() -> None:
    prepare_environment()
    port = free_port()
    SERVER_FILE.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
    threading.Thread(target=watch_and_open, args=(port,), daemon=True).start()

    os.chdir(APP_DIR)  # Streamlit reads .streamlit/config.toml from the working directory
    sys.path.insert(0, str(APP_DIR))
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(APP_DIR / "app.py"),
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--server.runOnSave", "false",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]
    try:
        stcli.main()
    except SystemExit:
        pass
    finally:
        SERVER_FILE.unlink(missing_ok=True)


def stop_server() -> None:
    try:
        info = json.loads(SERVER_FILE.read_text(encoding="utf-8"))
        pid = int(info["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if pid != os.getpid():
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,  # noqa: S603, S607
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    SERVER_FILE.unlink(missing_ok=True)


def main() -> None:
    (STATE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=STATE_DIR / "logs" / "launcher.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if sys.stdout is None:  # pythonw.exe: give libraries that print somewhere harmless to write
        sys.stdout = sys.stderr = open(STATE_DIR / "logs" / "server-output.log", "a", encoding="utf-8")  # noqa: SIM115

    if "--stop" in sys.argv[1:]:
        stop_server()
        return
    info = running_server()
    if info:
        open_window(int(info["port"]))
        return
    if not take_mutex():  # another launcher is starting the server right now
        info = wait_for_server(120)
        if info:
            open_window(int(info["port"]))
        return
    log.info("Starting DocIntel from %s", APP_DIR)
    run_server()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - last resort: tell the user instead of failing silently
        log.exception("Launcher failed")
        message_box(f"DocIntel could not start:\n{exc}\n\nDetails: {LOG_HINT}")
        sys.exit(1)
