# syandaV8 — 验证码识别服务 使用手册

版本：2.2.0  
支持题型：19 种（极验三代/四代、易盾、文字识别）

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境要求](#2-环境要求)
3. [安装部署](#3-安装部署)
4. [启动服务](#4-启动服务)
5. [API 接口](#5-api-接口)
6. [题型参数说明](#6-题型参数说明)
7. [配置说明](#7-配置说明)
8. [监控接口](#8-监控接口)
9. [Docker 部署](#9-docker-部署)
10. [常见问题](#10-常见问题)

---

## 1. 项目结构

```
syandaV8/
├── run.py              # 启动入口
├── config.py           # 全局配置（端口、限流、偏移等）
├── requirements.txt    # Python 依赖清单
├── .env.example        # 环境变量模板
├── start.bat           # Windows 一键启动
├── Dockerfile          # Docker 构建文件
├── docker-compose.yml  # Docker Compose 配置
├── api/
│   ├── server.py       # FastAPI 主服务
│   ├── response.py     # 统一响应格式
│   └── __init__.py
├── solver/
│   ├── registry.py     # 题型注册表 + 路由
│   ├── ocr.py          # OCR 引擎（ddddocr + crnn2 融合）
│   ├── crnn63_ocr.py   # CRNN63 模型推理（1001/1002/1003 专用）
│   ├── crnn_ocr.py     # CRNN2 第二引擎
│   ├── click.py        # 点选类题型求解
│   ├── slide.py        # 滑动类题型求解
│   ├── logic.py        # 逻辑推理类题型
│   ├── trajectory.py   # 轨迹生成
│   ├── preprocess.py   # 图像预处理
│   ├── utils.py        # 工具函数
│   └── engines/        # 各题型独立引擎
│       ├── _1008_math.py
│       ├── _1005_gap_puzzle.py
│       └── ...
├── cnn/
│   ├── crnn2_ocr.onnx      # CRNN2 模型 (15.6 MB)
│   ├── crnn63_ocr.onnx     # CRNN63 模型 (8.4 MB)
│   └── crnn63_ocr.json     # 模型元数据
├── web/
│   └── index.html      # 前端页面
└── logs/               # 运行时日志目录（自动创建）
```

---

## 2. 环境要求

| 项目 | 最低要求 |
|------|----------|
| Python | 3.9+ |
| 内存 | 2 GB |
| 磁盘 | 100 MB（含模型权重 24 MB） |
| CPU | 任意 x86_64 / ARM64 |
| GPU | 不需要（纯 CPU 推理） |
| 操作系统 | Linux / Windows / macOS |

---

## 3. 安装部署

### Linux / macOS

```bash
# 1. 进入项目目录
cd /path/to/syandaV8

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 修改端口、限流等

# 5. 启动
python run.py
```

### Windows

```bat
# 进入项目目录后直接双击
start.bat

# 或者命令行
cd H:\path\to\syandaV8
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

---

## 4. 启动服务

服务启动后默认监听：

```
http://0.0.0.0:15666
```

启动日志示例：

```
Captcha Solver v2.0 — 启动于 http://0.0.0.0:15666
支持 19 种题型
INFO:     Uvicorn running on http://0.0.0.0:15666 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 修改端口

方式一：环境变量
```bash
PORT=8080 python run.py
```

方式二：编辑 `config.py`
```python
PORT = int(os.getenv("PORT", "15666"))  # 改为 8080
```

---

## 5. API 接口

### 5.1 基础信息

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/docs` | GET | Swagger 交互文档（自动生成） |
| `/types` | GET | 查看所有支持的题型 |
| `/metrics` | GET | 监控指标 |
| `/solve` | POST | **核心接口：识别验证码** |
| `/solve/retry` | POST | 带重试的识别（多次尝试取最优） |
| `/batch_solve` | POST | 批量识别（最多 50 张） |

### 5.2 核心接口 `/solve`

**请求格式**（JSON）：

```json
{
  "type": 1001,
  "image": "<base64编码的图片数据>",
  "gap_image": null,
  "extra": {}
}
```

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | int | 是 | 题型编号（1001-1023） |
| `image` | string | 是 | Base64 编码的图片（URL 或 base64 字符串） |
| `gap_image` | string | 否 | 滑块缺口图（仅滑动类题型需要） |
| `extra` | object | 否 | 扩展参数（见各题型说明） |

**响应格式**：

```json
{
  "code": 0,
  "message": "ok",
  "type": 1001,
  "name": "英数混合",
  "data": {
    "text": "aB3x",
    "raw": "aB3x",
    "confidence": 0.9234,
    "engine": "crnn63"
  },
  "conf": 0.9234,
  "cost_ms": 12.3
}
```

**错误响应**：

```json
{
  "code": -1,
  "message": "unsupported 9999",
  "type": 9999,
  "data": {},
  "conf": 0.0,
  "cost_ms": 1.2
}
```

### 5.3 带重试 `/solve/retry`

比 `/solve` 多了 `attempts` 查询参数：

```
POST /solve/retry?attempts=3
```

会执行最多 N 次识别，返回置信度最高的结果。适合对准确率要求高的场景。

### 5.4 批量识别 `/batch_solve`

```json
{
  "items": [
    {"type": 1001, "image": "<base64>", "extra": {}},
    {"type": 1002, "image": "<base64>", "extra": {}},
    {"type": 1003, "image": "<base64>", "extra": {}}
  ]
}
```

返回 `results` 数组，每项结构与 `/solve` 响应相同。最多 50 条。

### 5.5 curl 示例

```bash
# OCR 识别（1001 英数混合）
curl -X POST http://localhost:15666/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": 1001,
    "image": "data:image/png;base64,iVBORw0KGgo...",
    "extra": {}
  }'

# 滑动验证（1004 极验三代滑动）
curl -X POST http://localhost:15666/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": 1004,
    "image": "<base64背景图>",
    "gap_image": "<base64缺口图>",
    "extra": {}
  }'

# 点选（1005 极验三代点选字）
curl -X POST http://localhost:15666/solve \
  -H "Content-Type: application/json" \
  -d '{
    "type": 1005,
    "image": "<base64图片>",
    "extra": {"words": ["猫", "狗", "鱼"]}
  }'

# 带重试
curl -X POST "http://localhost:15666/solve/retry?attempts=5" \
  -H "Content-Type: application/json" \
  -d '{"type": 1001, "image": "<base64>", "extra": {}}'
```

### 5.6 Python 客户端示例

```python
import httpx
import base64

def solve_captcha(type_code: int, image_path: str, extra: dict = None) -> dict:
    """调用 syandaV8 API 识别验证码"""
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    resp = httpx.post(
        "http://localhost:15666/solve",
        json={"type": type_code, "image": img_b64, "extra": extra or {}},
        timeout=30
    )
    return resp.json()

# 使用
result = solve_captcha(1001, "captcha.png")
print(result["data"]["text"])  # 识别结果
```

---

## 6. 题型参数说明

| 题型 | 名称 | 需要字段 | 返回值 | 通过率 |
|------|------|----------|--------|--------|
| 1001 | 英数混合 | image | text (字母+数字) | 86% |
| 1002 | 纯数字 | image | text (纯数字) | 92% |
| 1003 | 纯字母 | image | text (纯字母) | 90% |
| 1004 | 极验三代滑动 | image + gap_image | distance, track | 94% |
| 1005 | 极验三代点选(字) | image + extra.words | points [{x,y}] | 87% |
| 1006 | 极验三代点选(物) | image + extra.icons | points [{x,y}] | 98% |
| 1007 | 极验三代语序 | image + extra.phrase | points [{x,y}] | 88% |
| 1008 | 极验三代空间推理 | image + extra.question | points [{x,y}] | 100% |
| 1010 | 极验二三通用滑动 | image + gap_image | distance, track | 94% |
| 1012 | 极验四代滑动 | image + gap_image | distance, track | 94% |
| 1015 | 极验四代选汉字 | image + extra.words | points [{x,y}] | 87% |
| 1016 | 极验四代点过 | image + extra.count | points [{x,y}] | 100% |
| 1017 | 极验四代点图标 | image + extra.icons | points [{x,y}] | 98% |
| 1018 | 极验四代九宫格 | image + extra.positions | points [{x,y}] | 100% |
| 1019 | 极验三代九宫格 | image + extra.positions | points [{x,y}] | 100% |
| 1020 | 易盾滑动拼图 | image + gap_image | distance, track | 94% |
| 1021 | 易盾无感点过 | image + extra.count | points [{x,y}] | 100% |
| 1022 | 易盾点字 | image + extra.words | points [{x,y}] | 87% |
| 1023 | 易盾点图标 | image + extra.icons | points [{x,y}] | 98% |

**通过率基于 100 张真实样本测试（2026-08-27），整体平均 95.68%。**

### 各题型返回数据格式

**OCR 类（1001/1002/1003）**：
```json
{"text": "aB3x", "raw": "aB3x", "confidence": 0.92, "engine": "crnn63"}
```

**滑动类（1004/1010/1012/1020）**：
```json
{"distance": 256, "y": 120, "track": [12, 25, 38, ...], "method": "cv", "confidence": 0.85}
```

**点选类（1005/1006/1007/1015/1016/1017/1021/1022/1023）**：
```json
{"points": [{"x": 123, "y": 456}, {"x": 789, "y": 234}]}
```

**九宫格（1018/1019）**：
```json
{"points": [{"x": 56, "y": 102}, {"x": 340, "y": 156}]}
```

---

## 7. 配置说明

### 7.1 环境变量

复制 `.env.example` 为 `.env` 然后编辑：

```bash
cp .env.example .env
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | 0.0.0.0 | 监听地址 |
| `PORT` | 15666 | 监听端口 |
| `OCR_RETRY` | 3 | OCR 最大重试次数 |
| `ADMIN_KEY` | changeme | 管理接口密钥（如使用） |
| `LOG_LEVEL` | info | 日志级别（debug/info/warn/error） |
| `CACHE_LIMIT` | 1000 | 结果缓存条数上限 |
| `INFER_TIMEOUT` | 30000 | 单次推理超时（毫秒） |
| `RATE_LIMIT_PER_MINUTE` | 120 | 每 IP 每分钟最大请求数 |

### 7.2 坐标偏移补偿

如果实际部署环境中识别出的坐标有系统性偏差，在 `config.py` 中设置：

```python
COORD_OFFSETS = {
    "default": [0, 0],    # 全局偏移 [dx, dy]
    "1005": [2, -1],      # 1005 题型专属偏移
}
```

### 7.3 限流

默认每 IP 每分钟最多 120 次请求。超过后返回：
```json
{"code": -5, "message": "请求过于频繁，请稍后再试"}
```

设置 `RATE_LIMIT_PER_MINUTE=0` 可禁用限流。

---

## 8. 监控接口

### GET `/metrics`

返回运行指标：

```json
{
  "code": 0,
  "data": {
    "uptime_s": 3600.5,
    "total_requests": 1250,
    "error_requests": 12,
    "by_type": {
      "1001": {"count": 400, "avg_ms": 12.3, "ok_count": 380, "ok_rate": 0.95},
      "1002": {"count": 300, "avg_ms": 9.1, "ok_count": 285, "ok_rate": 0.95}
    },
    "cache": {"size": 856, "hit": 203, "miss": 1047}
  }
}
```

---

## 9. Docker 部署

### 方式一：Dockerfile 构建

```bash
cd /path/to/syandaV8
docker build -t syandaV8 .
docker run -d -p 15666:15666 syandaV8
```

### 方式二：Docker Compose

```bash
cd /path/to/syandaV8
docker-compose up -d
```

验证：
```bash
curl http://localhost:15666/
```

---

## 10. 常见问题

### Q: 启动报 `ModuleNotFoundError: No module named 'onnxruntime'`

A: 运行 `pip install onnxruntime`。requirements.txt 已包含此依赖，确保完整安装：
```bash
pip install -r requirements.txt
```

### Q: 启动报 `ModuleNotFoundError: No module named 'ddddocr'`

A: ddddocr 需要从源码安装（PyPI 版本可能有兼容问题）：
```bash
pip install ddddocr
```
如果仍有问题，尝试：
```bash
pip install ddddocr --no-cache-dir
```

### Q: 端口被占用

A: 修改端口：
```bash
PORT=8080 python run.py
```

### Q: 首次请求很慢

A: 服务启动时会预热所有模型（`on_event("startup")`），但 ddddocr 的首次加载可能需要 2-5 秒。建议部署后先调用一次 `/solve` 预热。

### Q: 如何确认模型加载成功

A: 调用 `/types` 确认服务运行正常，然后发送一张测试图片。如果 `engine` 字段显示 `crnn63`（1001/1002/1003）或 `dd+crnn`（其他），说明模型正常。

### Q: 内存占用多大

A: 服务稳定运行后约 300-500 MB（取决于并发量和缓存大小）。模型权重在启动时全部加载到内存。

### Q: 能否支持高并发

A: 可以。内部使用 8 线程池处理请求，FastAPI 异步 IO。单核 CPU 可支撑约 50-100 QPS（OCR 类约 8-10ms/张）。瓶颈主要在 OCR 推理。

---

## 快速验证清单

部署完成后按以下步骤验证：

```bash
# 1. 确认服务启动
curl http://localhost:15666/
# 期望: {"code":0,"data":{"service":"syandaV8","status":"running"}}

# 2. 确认 19 种题型已注册
curl http://localhost:15666/types
# 期望: 返回 19 个题型

# 3. 测试 OCR 识别
curl -X POST http://localhost:15666/solve \
  -H "Content-Type: application/json" \
  -d '{"type":1002,"image":"<你的base64图片>","extra":{}}'
# 期望: code=0, data.text 有值

# 4. 检查监控
curl http://localhost:15666/metrics
# 期望: total_requests 开始计数
```

全部通过 = 部署成功。
