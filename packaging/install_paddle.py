"""Optional add-on for the installed DocIntel app: download and install the PaddleOCR engine.

Started from Start menu -> "DocIntel - Install PaddleOCR" or from the app's Settings page.
Runs in a console window with the app's own Python:
    python.exe install_paddle.py            install (asks for confirmation)
    python.exe install_paddle.py --yes      install without asking

Steps: stop a running DocIntel (some files cannot be replaced while it runs), install the
pinned PaddlePaddle/PaddleOCR packages into the app's Python, download the recognition models
with one test OCR, then restart DocIntel. Needs internet once (about 700 MB download).
Updating DocIntel replaces its Python folder; run this again afterwards to get PaddleOCR back.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PYTHON_DIR = Path(sys.executable).resolve().parent
PYTHONW = PYTHON_DIR / "pythonw.exe"
LAUNCHER = APP_DIR / "DocIntel.pyw"
REQUIREMENTS = APP_DIR / "requirements-paddle.txt"
CONSTRAINTS = APP_DIR / "paddle-constraints.txt"   # versions tested together (written by the build)


def say(text: str = "") -> None:
    print(text, flush=True)


def finish(code: int) -> None:
    if "--yes" not in sys.argv:
        input("\nPress Enter to close this window...")
    sys.exit(code)


def stop_app() -> None:
    if PYTHONW.is_file() and LAUNCHER.is_file():
        subprocess.run([str(PYTHONW), str(LAUNCHER), "--stop"], check=False)  # noqa: S603
        time.sleep(2)


def start_app() -> None:
    if PYTHONW.is_file() and LAUNCHER.is_file():
        subprocess.Popen([str(PYTHONW), str(LAUNCHER)], cwd=str(APP_DIR), close_fds=True)  # noqa: S603


def install() -> bool:
    args = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location",
            "--only-binary=:all:", "-r", str(REQUIREMENTS), "setuptools"]  # paddle imports setuptools
    if CONSTRAINTS.is_file():
        args += ["-c", str(CONSTRAINTS)]
    say("Installing packages (several minutes; progress below)...\n")
    return subprocess.run(args, check=False).returncode == 0  # noqa: S603


def download_models() -> bool:
    """One small OCR run: PaddleOCR downloads its English models (~16 MB) on first use."""
    say("\nDownloading the PaddleOCR models and running a test...")
    code = (
        "from docintel.config.settings import Settings\n"
        "from docintel.evaluation.synthetic_data import render_text_image\n"
        "from docintel.ocr.base_engine import OCROptions\n"
        "from docintel.ocr.engine_factory import EngineFactory\n"
        "f = EngineFactory(Settings(enable_paddle=True, log_to_file=False))\n"
        "r = f.get('paddle').recognize(render_text_image(['PaddleOCR test 2024']),"
        " OCROptions(languages=('en',), psm=None, timeout_seconds=600))\n"
        "print('Test result:', r.text.strip() or '(empty)')\n"
        "f.close_all()\n"
    )
    env = dict(os.environ, DOCINTEL_ENABLE_PADDLE="true")
    return subprocess.run([sys.executable, "-c", code], cwd=str(APP_DIR), env=env, check=False).returncode == 0  # noqa: S603


def main() -> None:
    os.chdir(APP_DIR)
    say("DocIntel - install the optional PaddleOCR engine")
    say("=" * 50)
    say("PaddleOCR is a deep-learning OCR engine. It can read phone photos and difficult")
    say("images better than Tesseract, but is slower and uses 1-2 GB more memory.")
    say("")
    say("* Download: about 700 MB (internet needed once); about 1.3 GB on disk")
    say("* DocIntel closes while installing and reopens afterwards")
    say("* Afterwards choose 'PaddleOCR' as the engine in DocIntel")
    say("")
    if not REQUIREMENTS.is_file():
        say(f"Missing {REQUIREMENTS.name}; reinstall DocIntel.")
        finish(1)
    if "--yes" not in sys.argv and input("Install PaddleOCR now? [y/N] ").strip().lower() not in ("y", "yes"):
        say("Nothing was changed.")
        finish(0)

    say("\nClosing DocIntel...")
    stop_app()
    if not install():
        say("\nInstallation FAILED (see the messages above). Check the internet connection and try again.")
        say("DocIntel still works with Tesseract.")
        start_app()
        finish(1)
    if not download_models():
        say("\nPaddleOCR is installed, but the model download or test failed.")
        say("It will retry automatically the first time you use PaddleOCR (needs internet).")
    else:
        say("\nPaddleOCR is installed and working.")
    say("Starting DocIntel...")
    start_app()
    finish(0)


if __name__ == "__main__":
    main()
