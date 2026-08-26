# syandaV8 v2.1

## 验证码识别服务 — 19 种题型，HTTP API 方式调用。

全部存储跑 H 盘，不占 C 盘。

## 快速启动

```bash
cd H:\qinglong\syandaV8
.venv\Scripts\python.exe run.py
```

服务启动在 `http://0.0.0.0:8000`

## API 文档

启动后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 统一求解接口

```
POST /solve
Content-Type: application/json

{
  "type": 1004,          // 题型编号
  "image": "base64...",  // 图片 base64
  "gap_image": null,     // 滑块小图（滑块类需要）
  "extra": {}            // 额外参数
}
```

返回：
```json
{
  "code": 0,
  "message": "ok",
  "type": 1004,
  "name": "极验-三代滑动",
  "data": {
    "distance": 128,
    "y": 97,
    "track": [...],
    "method": "tmpl",
    "confidence": 0.9732
  },
  "conf": 0.9732,
  "cost_ms": 123.4
}
```

### 带重试求解

```
POST /solve/retry?attempts=3
```

对同一张图片做多次求解（滑块多尺度），返回置信度最高的结果。

### 兼容打码狗接口

```
POST /apiv1/ocr   (form: image, type)
POST /apiv1/slide (form: bg, gap)
```
## 题型列表 (19 种)

| 编号 | 名称 | 引擎 |
|------|------|------|
| 1001 | 英数混合 | OCR 多预处理+投票 |
| 1002 | 纯数字 | OCR |
| 1003 | 纯字母 | OCR |
| 1004 | 极验三代滑动 | alpha-mask 模板匹配 |
| 1005 | 三代点选选字 | syandaV8 检测 |
| 1006 | 三代点选选物 | 模板匹配 |
| 1007 | 三代语序选择 | 文字排序 |
| 1008 | 三代空间推理 | 空间排序 |
| 1010 | 极验通用滑动 | alpha-mask |
| 1012 | 极验四代滑动 | alpha-mask |
| 1015 | 四代选汉字 | syandaV8 检测 |
| 1016 | 四代点过 | 检测+排序 |
| 1017 | 四代点图标 | 模板匹配 |
| 1018 | 四代九宫格 | 网格计算 |
| 1019 | 三代九宫格 | 网格计算 |
| 1020 | 易盾滑动拼图 | alpha-mask |
| 1021 | 易盾无感点过 | 检测+排序 |
| 1022 | 易盾点字 | syandaV8 检测 |
| 1023 | 易盾点图标 | 模板匹配 |

## 通过率

- 滑块（韵达实测）：单次 90-93%，3 次重试后 ~99.9%
- OCR（合成 200 张）：单次 85%## 部署到服务器

```bash
cd /opt/syandaV8
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
PORT=8000 nohup python run.py > logs/server.log 2>&1 &
```

- 全部缓存走 `__cache/`，模型走 `__cache/models`，不占系统盘
- 日志写 `logs/`
- 端口/密钥用环境变量 `PORT`、`ADMIN_KEY` 覆盖

## 客户端调用示例

见 `scripts/client_retry.py`（通用重试模板）、`scripts/yunda_client.py`（韵达生产示例）。
