# syandaV8 v2.2 -- 独立部署指南

> 本项目完全自包含，复制到任何服务器即可直接运行，无需修改代码。

## 目录

1. [项目完整性清单](#1)
2. [系统要求](#2)
3. [Linux/macOS 快速部署](#3)
4. [Windows 部署](#4)
5. [Docker 部署](#5)
6. [环境变量配置](#6)
7. [验证部署](#7)
8. [故障排查](#8)
9. [性能参考](#9)

---

## 1. 项目完整性清单

`
syandaV8/
├── run.py                    # 启动入口 (764B)
├── config.py                 # 全局配置 (1388B)
├── requirements.txt          # Python 依赖 (226B)
├── .env.example              # 环境变量模板 (322B)
├── start.bat                 # Windows 一键启动
├── Dockerfile                # Docker 构建文件
├── docker-compose.yml        # Docker Compose
├── README.md                 # 项目说明
├── USAGE.md                  # 使用手册
│
├── api/                      # API 服务层
│   ├── server.py             # FastAPI 主服务 (9292B)
│   └── response.py           # 统一响应格式 (848B)
│
├── solver/                   # 求解引擎
│   ├── registry.py           # 题型注册表+路由
│   ├── ocr.py                # OCR 引擎 (双模型)
│   ├── click.py              # 点选求解 (12674B)
│   ├── slide.py              # 滑动求解
│   ├── logic.py / trajectory.py
│   ├── preprocess.py         # 图像预处理
│   └── engines/              # 9个独立题型引擎
│
└── cnn/                      # ONNX 模型 (~24MB)
    ├── crnn2_ocr.onnx        # CRNN2 模型 (15.6MB)
    ├── crnn63_ocr.onnx       # CRNN63 模型 (8.4MB)
    └── crnn63_ocr.json       # 词表映射
`

**关键特性：**
- 所有路径基于 __file__ 动态锚定，无外部硬编码路径
- 模型文件内嵌在 cnn/ 目录中
- 可放在任意位置，无需修改任何代码
- 全部缓存走 __cache/，不占系统盘

---

## 2. 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux (Ubuntu 20+/Debian 11+), macOS 12+, Windows Server 2016+ |
| Python | >= 3.10 (推荐 3.10-3.12) |
| 内存 | >= 2 GB RAM |
| 磁盘 | >= 500 MB (模型 + 运行时缓存) |
| CPU | x86_64 或 ARM64 |
| GPU | 不需要，纯CPU推理 |

---

## 3. Linux/macOS 快速部署

`ash
# 上传到服务器
scp -r syandaV8/ root@your-server:/opt/

# SSH登录并进入目录
cd /opt/syandaV8

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖（国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置（可选）
cp .env.example .env

# 启动服务
python run.py               # 前台启动
# nohup python run.py > logs/server.log 2>&1 &  # 后台

# 验证
curl http://localhost:15666/
`

**预期返回：**
`json
{"code":0,"message":"ok","data":{"service":"syandaV8","version":"2.2.0","types":19,"status":"running"}}
`

---

## 4. Windows 部署

### 方式一：命令行

`powershell
cd H:\syandaV8
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
`

### 方式二：一键启动

双击运行 start.bat，自动创建venv、安装依赖、启动服务。

---

## 5. Docker 部署

`ash
cd /opt/syandaV8
docker compose up -d --build
docker compose logs -f
curl http://localhost:15666/
`

常用命令：
| 命令 | 作用 |
|------|------|
| docker compose down | 停止服务 |
| docker compose restart | 重启服务 |
| docker compose ps | 查看状态 |

默认内存限制 2GB，可按需修改 docker-compose.yml。

---

## 6. 环境变量配置

复制 .env.example 为 .env 后编辑：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 15666 | 服务端口 |
| OCR_RETRY | 3 | OCR重试次数 |
| ADMIN_KEY | changeme | API管理密钥 |
| LOG_LEVEL | info | 日志级别 |
| CACHE_LIMIT | 1000 | 缓存条目上限 |
| INFER_TIMEOUT | 30000 | 推理超时毫秒 |
| RATE_LIMIT_PER_MINUTE | 120 | 每分钟请求限制(0=不限) |

---

## 7. 验证部署

`ash
curl http://localhost:15666/           # 健康检查
curl http://localhost:15666/types     # 题型列表（应有19种）
curl http://localhost:15666/metrics   # 监控指标
`

---

## 8. 故障排查

### ImportError / ModuleNotFoundError
- 确认已激活虚拟环境：source .venv/bin/activate
- 重新安装：pip install -r requirements.txt

### 端口被占用
- 修改 .env 中的 PORT 为其他端口
- 或 kill 占用进程

### 模型加载失败
- 确认 cnn/ 目录下有两个 .onnx 文件（总大小约 24MB）
- 检查 onnxruntime：python -c "import onnxruntime; print(onnxruntime.__version__)" (>=1.15.0)

### libgomp/libgtk 缺失 (Linux Docker)
`ash
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgbm1 libgomp1
`

### 内存不足 (< 2GB)
- 设置 SWAP：allocate -l 2G /swapfile && swapon /swapfile

---

## 9. 性能参考

| 场景 | 延迟 | 并发 |
|------|------|------|
| OCR 识别 (单次) | 30-100ms | 100+ QPS |
| 滑动识别 (单次) | 50-200ms | 50+ QPS |
| 点选识别 (单次) | 80-300ms | 30+ QPS |
| 批量识别 (50张) | 2-5s | 取决于并发 |

*版本：v2.2.0 | 支持题型：19种 | 最后更新：2026-08-27*