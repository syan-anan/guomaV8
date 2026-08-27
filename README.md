# guomaV8 — 验证码识别服务

**当前版本：D4** ｜ **19 种题型全覆盖** ｜ Docker 一键部署 ｜ HTTP API 调用

---

## ⚡ 一键部署（推荐）

只需一条命令，自动完成克隆、构建、启动：

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/syan-anan/guomaV8/main/deploy.sh)
```

> 国内网络不通时使用加速地址下载脚本：
>
> ```bash
> bash <(curl -fsSL https://ghproxy.net/https://raw.githubusercontent.com/syan-anan/guomaV8/main/deploy.sh)
> ```

可选环境变量：

```bash
PORT=15666 APP_DIR=/opt/guomaV8 bash <(curl -fsSL https://raw.githubusercontent.com/syan-anan/guomaV8/main/deploy.sh)
```

部署完成后验证：

```bash
curl http://127.0.0.1:15666/health
```

---

## 🐳 手动 Docker 部署

### 方式一：docker run（和青龙面板同风格）

```bash
git clone https://github.com/syan-anan/guomaV8.git
cd guomaV8
cp .env.example .env
docker build -t guoma-v8:latest .
docker run -d \\
  --name guomaV8 \\
  --restart always \\
  -p 15666:15666 \\
  -v ./logs:/app/logs \\
  guoma-v8:latest
```

### 方式二：docker compose

```bash
git clone https://github.com/syan-anan/guomaV8.git
cd guomaV8
cp .env.example .env
docker compose up -d
```

### 常用管理命令

| 操作 | 命令 |
|------|------|
| 查看日志 | `docker logs -f guomaV8` |
| 重启 | `docker restart guomaV8` |
| 停止 | `docker stop guomaV8 && docker rm guomaV8` |
| 更新版本 | `cd /opt/guomaV8 && git pull && docker compose up -d --build` |

---

## 🔌 API 使用教程

服务端口：`15666`，认证方式：Header 带 `X-Admin-Key`（密钥在 `.env` 的 `ADMIN_KEY`）。

### 统一求解接口 POST /solve

```bash
curl -X POST http://127.0.0.1:15666/solve \\
  -H "Content-Type: application/json" \\
  -H "X-Admin-Key: 你的ADMIN_KEY" \\
  -d '{
    "type": 1004,
    "image": "背景图base64",
    "gap_image": "滑块小图base64"
  }'
```

返回示例：

```json
{ "code": 0, "data": { "x": 186, "y": 34 }, "cost_ms": 42 }
```

### 支持的题型（19 种）

| 分组 | 题型编号 |
|------|----------|
| 普通 OCR | 1001 英数混合 · 1002 纯数字 · 1003 纯字母 |
| 极验三代 | 1004 滑动 · 1005 点选选字 · 1006 点选选物 · 1007 语序 · 1008 空间推理 · 1019 九宫格 |
| 极验四代 | 1012 滑动 · 1013 五子棋 · 1014 消消乐 · 1015 选汉字 · 1016 点过 · 1017 点图标 · 1018 九宫格 |
| 易盾 | 1020 滑动拼图 · 1021 无感点过 · 1022 点字 · 1023 点图标 |
| 其他 | 1009 极验无感知 · 1011 谷歌 recaptcha |

### 文档与调试

- Swagger 在线文档：`http://127.0.0.1:15666/docs`
- 服务自检：`python scripts/service_selftest_19types.py`
- 健康检查：`GET /health`

---

## ⚙️ 配置说明（.env）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 15666 | 服务端口 |
| ADMIN_KEY | （必改） | 接口鉴权密钥 |
| OCR_RETRY | 3 | 识别重试次数 |
| RATE_LIMIT_PER_MINUTE | 1000 | 每分钟请求上限 |
| INFER_TIMEOUT | 30000 | 单次推理超时(ms) |

---

## 📜 版本历史

- **D4（当前）**：模型再优化 + 一键部署脚本
- **D3**：CRNN63 双模型融合
- **D2 / D1**：初版引擎与基准测试

## License

仅供学习研究使用。
