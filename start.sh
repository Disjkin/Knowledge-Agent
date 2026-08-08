#!/usr/bin/env bash
# 启动知识库助手（Linux / macOS / Git Bash）
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "[提示] .env 不存在，已从 .env.example 复制一份，请先编辑填入 API key。"
  cp .env.example .env
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
else
  exec python -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
fi
