#!/usr/bin/env bash
# guomaV8 一键部署脚本（Linux / macOS / 支持 Docker 的 NAS）
# 用法: bash <(curl -fsSL https://cdn.jsdelivr.net/gh/syan-anan/guomaV8@main/deploy.sh)
set -e

APP_DIR=${APP_DIR:-/opt/guomaV8}
PORT=${PORT:-15666}
REPO=https://github.com/syan-anan/guomaV8.git
MIRROR=https://ghproxy.net/${REPO}

echo "==> guomaV8 一键部署开始"
command -v docker >/dev/null 2>&1 || { echo "[错误] 未检测到 Docker，请先安装: https://docs.docker.com/get-docker/"; exit 1; }

if [ ! -d "$APP_DIR/.git" ]; then
    echo "==> 克隆仓库到 $APP_DIR"
    git clone --depth 1 $REPO $APP_DIR 2>/dev/null || git clone --depth 1 $MIRROR $APP_DIR
else
    echo "==> 更新已有仓库"
    cd $APP_DIR && git pull --ff-only || true
fi

cd $APP_DIR
[ -f .env ] || cp .env.example .env

echo "==> 构建镜像（首次约 2-5 分钟）"
docker build -t guoma-v8:latest .

echo "==> 启动容器"
docker rm -f guomaV8 2>/dev/null || true
docker run -d \\
  --name guomaV8 \\
  --restart always \\
  -p ${PORT}:15666 \\
  -v "$PWD/logs:/app/logs" \\
  guoma-v8:latest

sleep 3
STATUS=$(curl -s http://127.0.0.1:${PORT}/health || echo "(本机无 curl 可忽略)")
echo "==> 部署完成！服务地址: http://服务器IP:${PORT}"
echo "==> 健康检查响应: $STATUS"
echo "==> 管理密钥在 $APP_DIR/.env 的 ADMIN_KEY 中，请及时修改"
