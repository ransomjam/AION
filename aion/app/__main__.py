"""Launch the AION application:  python -m aion.app

    python -m aion.app                 # 127.0.0.1:7071
    python -m aion.app --port 8080
    AION_PORT=9000 python -m aion.app  # via environment
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser

from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aion.app",
                                     description="AION operating environment")
    parser.add_argument("--host", default=os.environ.get("AION_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AION_PORT", "7071")))
    parser.add_argument("--no-browser", action="store_true",
                        help="do not attempt to open a browser window")
    args = parser.parse_args(argv)

    if not args.no_browser and args.port != 0:
        url = f"http://{args.host}:{args.port}/"
        # Open after a short delay so the server is listening first. Best-effort:
        # headless environments simply won't have a browser to open.
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
