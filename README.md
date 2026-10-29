# syandmV8 D5

**版本：D5** ｜ 通用打码服务 + reCAPTCHA 统一入口 ｜ 可独立运行

## 包含两条线

1. **打码 API**（原 D4 能力，25 种题型）：`run.py` → `http://0.0.0.0:15666`
2. **reCAPTCHA 求解**（D5 新增）：`tools/rc_solve_auto.py`，图像先试、失败自动切音频

## 安装

需要 Python 3.10+，Windows 下另需本机 Chrome（reCAPTCHA 线使用 `channel="chrome"`）。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

模型全部随包提供，无需下载：

- `cnn/`：OCR / 路由 / 点选 / 特化模型（61 个运行期文件）
- `_bk_tmp/clipl/`：CLIP 视觉塔（图像九宫格打分）
- `_bk_tmp/whisper_tiny_en/`：音频转写
- `cnn/rc_tile/rc_head_da.onnx`：九宫格分类头（ONNX，无 torch 依赖）

## 启动打码 API

```bash
python run.py
```

默认 `HOST=0.0.0.0`、`PORT=15666`，可用环境变量覆盖。

```
GET  /            服务信息
GET  /types       题型列表
POST /solve       {"type":1001,"image":"<base64>"}
POST /apiv1/ocr   multipart(image,type)
```

## 启动 reCAPTCHA 求解

```bash
python tools\rc_solve_auto.py 10 run1
python tools\rc_solve_auto.py 10 run1 https://your-site.example/path
```

参数：`次数`、`标签`、`可选目标 URL`。结果写入 `reports/rc_auto_<标签>.jsonl`。

策略与依赖：

- 每次一条住宅代理（`tools` 内已内置提取接口，换代理需改 `API` 常量）
- 3×3 图像最多 2 次提交；16 格或词表外类别直接走音频
- `blocked / noform / noaudio` 自动换代理重试，最多 3 次
- 依赖：干净住宅代理、本机 Chrome；音频通过依赖会话可信度，不是识别正确率

## 验收数据（D5）

| 项 | 结果 |
|---|---|
| 通用 OCR 单图 | 91.6–96.0%（六集） |
| Turnstile（本机 IP + 真实 profile） | 7/7 served |
| reCAPTCHA 统一入口 | 28/30 = 93.3%（image 3 / audio 25） |
| reCAPTCHA 图像九宫格 | 单题 11.2%，未达标 |
| 音频兜底 | 19/20 = 95% |

## 目录

```
run.py  config.py  requirements.txt
api/                 FastAPI 服务
solver/              25 种题型求解器 + engines/
tools/               rc_solve_auto.py, rc_audio_solve.py
cnn/                 运行期模型
_bk_tmp/clipl/       CLIP 视觉塔
_bk_tmp/whisper_tiny_en/
reports/             运行输出（初始为空）
MANIFEST.json        构建清单
```

## 已知边界

- 图像九宫格是短板：单格 precision 0.90，目标是 0.98；D5 靠音频兜底。
- 两条线网络配置相反：reCAPTCHA 要住宅代理，Turnstile 要本机 IP + 真实 profile。
- 社区站 429 会阻断 Turnstile 连续测量，需间隔重试。
