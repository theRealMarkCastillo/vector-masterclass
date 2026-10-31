"""Render exported notebooks offline in an isolated, bounded headless Chrome process."""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops

from vector_masterclass.config import root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Artifact run directory name")
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    browser = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not browser.exists():
        raise RuntimeError(
            "This QA command requires local Google Chrome; it does not change course runtime requirements"
        )
    run = root() / "artifacts" / args.run
    if not run.is_dir() or run.resolve().parent != (root() / "artifacts").resolve():
        raise ValueError("Choose one existing artifact run directory")
    paths = sorted(run.glob("*/*.html"))
    if args.ids:
        paths = [p for p in paths if p.parent.name in args.ids]
    for path in paths:
        screenshot = path.parent / "rendered-html.png"
        log = path.parent / "chrome-qa.log"
        profile = root() / ".cache/chrome-qa" / path.parent.name
        profile.mkdir(parents=True, exist_ok=True)
        if screenshot.exists():
            screenshot.unlink()
        with log.open("w") as stream:
            process = subprocess.Popen(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--disable-background-networking",
                    "--disable-component-update",
                    "--disable-extensions",
                    "--no-default-browser-check",
                    "--host-resolver-rules=MAP * ~NOTFOUND",
                    f"--user-data-dir={profile}",
                    f"--screenshot={screenshot}",
                    "--window-size=1280,12000",
                    "--virtual-time-budget=3000",
                    path.as_uri(),
                ],
                stdout=stream,
                stderr=stream,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 25
                while (
                    time.monotonic() < deadline
                    and not screenshot.exists()
                    and process.poll() is None
                ):
                    time.sleep(0.2)
                if not screenshot.exists():
                    raise TimeoutError(f"Browser did not render {path}; inspect {log}")
                print(screenshot, flush=True)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        # Trim only blank browser canvas, preserving all rendered page content.
        with Image.open(screenshot) as original:
            rgb = original.convert("RGB")
            bounds = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
            if bounds and bounds[3] >= rgb.height - 10:
                raise RuntimeError(f"Page exceeds QA viewport; increase height and rerender {path}")
            if bounds:
                rgb.crop((0, 0, rgb.width, min(rgb.height, bounds[3] + 40))).save(screenshot)


if __name__ == "__main__":
    main()
