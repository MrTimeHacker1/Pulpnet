#!/usr/bin/env bash
# Wait for backends to be reachable, then exec the service command.
set -e

# --- Qdrant (server mode) ---
if [ -n "${QDRANT_URL}" ]; then
  echo "[entrypoint] Waiting for Qdrant at ${QDRANT_URL} ..."
  for _ in $(seq 1 90); do
    if curl -sf "${QDRANT_URL}/readyz" >/dev/null 2>&1 \
       || curl -sf "${QDRANT_URL}/healthz" >/dev/null 2>&1 \
       || curl -sf "${QDRANT_URL}/" >/dev/null 2>&1; then
      echo "[entrypoint] Qdrant is up."
      break
    fi
    sleep 2
  done
fi

# --- Postgres (best-effort; compose depends_on already gates on healthy) ---
if [[ "${DB_URL}" == postgres* ]]; then
  echo "[entrypoint] Waiting for Postgres ..."
  python - <<'PY' || true
import os, time
import sqlalchemy as sa
url = os.environ.get("DB_URL", "")
eng = sa.create_engine(url)
for _ in range(60):
    try:
        with eng.connect() as c:
            c.execute(sa.text("SELECT 1"))
        print("[entrypoint] Postgres is up.")
        break
    except Exception:
        time.sleep(2)
PY
fi

exec "$@"
