@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist .env (
  echo [提示] .env 不存在，已从 .env.example 复制一份，请先编辑填入 API key。
  copy .env.example .env
)

echo 启动知识库助手...
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
