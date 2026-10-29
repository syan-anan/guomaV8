# 服务器部署包（D5）

镜像：`captcha-solver:d5` — **179MB**（D4 为 1.39GB）

## 结构

镜像内只有：python:3.12-slim + libgomp1 + 本仓库代码。
以下两项以只读卷挂载，不打进镜像：

```
./cnn  -> /app/cnn        运行期 OCR 模型（约 650MB）
./venv -> /opt/venv       Python 依赖（ddddocr / onnxruntime / opencv 等）
```

## 首次部署

```bash
cd /root/syandaV8_d5

# 1. 创建 venv（与镜像同为 Python 3.12）
docker run --rm -v $PWD/venv:/opt/venv python:3.12-slim python -m venv /opt/venv

# 2. 安装依赖到 venv
docker run --rm -v $PWD/venv:/opt/venv -v $PWD/requirements.server.txt:/tmp/req.txt \
  python:3.12-slim sh -c '/opt/venv/bin/pip install --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r /tmp/req.txt'

# 3. 构建极简镜像并启动
docker build -f Dockerfile.slim5 -t captcha-solver:d5 .
docker compose up -d --no-build

curl http://127.0.0.1:15666/
```

`cnn/` 需与 /root/syandaV8_d5/cnn 保持一致（发布包外置）。

## 为什么这么小

| 内容 | 位置 |
|---|---|
| Python 解释器 + OS | 镜像（约 180MB） |
| 代码 | 镜像（<1MB） |
| 依赖包 | 宿主 venv，卷挂载 |
| OCR 模型 | 宿主 cnn，卷挂载 |

## 回滚

旧镜像已按要求清理，不再保留 d4-rollback。需要回滚时：
用 git tag `D4`（如存在）或上一版代码重新构建；数据只依赖 `cnn/` 与 venv。

## 依赖提醒

- venv 与镜像的 Python 版本必须一致（3.12），否则无法启动。
- 迁移机器时需同时带走 `venv/`、`cnn/`、`.env`。
