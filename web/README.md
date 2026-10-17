# web/

The web UI, served by `linux/server.py`. React + TypeScript, built with Vite.

- `/` (phone): take a photo, see the top-5 labels and the latency of each pipeline stage.
- `/screen` (big screen): every photo anyone takes, live, plus session stats and recent shots.

`dist/` is committed so the board doesn't need Node. **Run `npm run build` and commit `dist/` after every change.**

## Develop on a laptop

```bash
cd linux && .venv/bin/python server.py --cpu   # backend on :8000 (see linux/README.md)
cd web && npm install && npm run dev           # UI with hot reload, calls the backend above
```

Or skip `npm run dev`: run `npm run build`, then open `http://localhost:8000` (served by `server.py`).
