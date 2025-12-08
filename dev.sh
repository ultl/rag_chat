#!/usr/bin/env bash
set -euo pipefail

# Load env vars from .env into the current shell
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC2046
  . ./.env
  set +a
fi

FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
NEXT_PUBLIC_API_BASE="${NEXT_PUBLIC_API_BASE:-http://$FASTAPI_HOST:$FASTAPI_PORT}"

echo "Starting backend on $FASTAPI_HOST:$FASTAPI_PORT"
FASTAPI_CMD="fastapi run app/main.py --host $FASTAPI_HOST --port $FASTAPI_PORT"

echo "Starting frontend on $FRONTEND_HOST:$FRONTEND_PORT (API: $NEXT_PUBLIC_API_BASE)"
FRONTEND_CMD="cd frontend && HOST=$FRONTEND_HOST PORT=$FRONTEND_PORT NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE bun dev"

# Start both and forward Ctrl+C to clean up
trap 'echo "Stopping..."; kill 0' INT TERM

sh -c "$FASTAPI_CMD" &
BACK_PID=$!

sh -c "$FRONTEND_CMD" &
FRONT_PID=$!

wait $BACK_PID $FRONT_PID
