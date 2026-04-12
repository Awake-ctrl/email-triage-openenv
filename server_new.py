"""
server.py (root shim)
---------------------
The canonical server is now at server/app.py as required by openenv validate.
This shim is kept for backwards compatibility only.

Run the server with:
  python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
or via the registered entry point after `pip install -e .`:
  serve
"""
from server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
