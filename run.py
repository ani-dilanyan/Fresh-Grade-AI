"""Start FreshGrade AI and open it in the browser.

    python run.py            → http://localhost:8000
    python run.py --port 8080
"""
import argparse
import threading
import webbrowser

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(4.0, lambda: webbrowser.open(url)).start()
    print(f"\n  FreshGrade AI starting… open {url}  (press CTRL+C to stop)\n")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
