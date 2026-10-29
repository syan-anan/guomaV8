# 服务器部署包（D5）
目录说明：

- `Dockerfile.slim`：生产镜像，Python 3.12-slim，只装 API 依赖，镜像 783MB
- `Dockerfile`：旧版完整依赖构建（1.12GB），仅作参考
- `requirements.server.txt`：服务端最小依赖（无 torch / patchright / whisper）
- `docker-compose.yml`：15666 端口，模型以只读卷挂载

## 模型不在镜像里

`cnn/` 通过卷挂载到 `/app/cnn`，镜像本身不含模型：

```
-v /root/syandaV8_d5/cnn:/app/cnn:ro
```

镜像因此从 1.39GB（D4）降到 **783MB**，功能不变（25 种题型）。

## 构建与启动

```bash
docker build -f Dockerfile.slim -t captcha-solver:d5 .
docker compose up -d --no-build
curl http://127.0.0.1:15666/
```

## 回滚

```bash
docker tag captcha-solver:d4-rollback captcha-solver:latest
cd /root/syandaV8 && docker compose up -d --no-build
```
