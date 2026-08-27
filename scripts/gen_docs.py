# gen_docs.py
# Generates DEPLOY.md and DEPLOY_AI.md for syandaV8
import pathlib

root = pathlib.Path(__file__).parent.parent

# ============ DEPLOY.md ============
deploy_md = """\
# syandaV8 Deployment Guide

> This project is fully self-contained in a single folder.
> Copy `syandaV8/` to any Linux/Windows server and run it directly. No external dependencies.

## Prerequisites

- **Python** >= 3.10 (tested with 3.14.6)
- **RAM** >= 2 GB available
- **Disk** >= 500 MB (ONNX models ~24 MB + runtime cache)

## Linux / macOS Quick Deploy

```bash
cd syandaV8
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit HOST / PORT / ADMIN_KEY
python run.py           # listens on http://0.0.0.0:15666
```

Background run:
```bash
nohup python run.py > logs/stdout.log 2>&1 &
echo $! > logs/run.pid
```

## Windows Quick Deploy

```powershell
cd H:\\qinglong\\syandaV8
python -m pip install -r requirements.txt
python run.py
# or double-click start.bat
```

## Docker Deploy (Optional)

```bash
docker build -t syandav8 .
docker run -d --name syandav8 -p 15666:15666 \\
  -v $(pwd)/logs:/app/logs \\
  -v $(pwd)/__cache:/app/__cache \\
  syandav8
```

## API Endpoints

| Method | Path         | Description                   |
|--------|--------------|-------------------------------|
| GET    | /            | Health check                  |
| GET    | /types       | List all 19 captcha types     |
| POST   | /solve       | Single image solve            |
| POST   | /batch_solve | Batch solve (max 50)          |
| GET    | /metrics     | Performance metrics           |
| GET    | /docs        | Swagger UI                    |

## File Inventory

```
syandaV8/
├── run.py                    # Entry point
├── config.py                 # Global config (.env loader)
├── requirements.txt          # 11 Python packages
├── .env.example              # Env var defaults
├── start.bat                 # Windows launcher
├── Dockerfile                # Docker build
├── docker-compose.yml        # Docker Compose
├── README.md                 # User docs
├── USAGE.md                  # Detailed usage tutorial
├── DEPLOY.md                 # This file
├── DEPLOY_AI.md              # AI-readable deployment guide
├── api/
│   ├── server.py             # FastAPI routes + rate limit + metrics
│   └── response.py           # Response wrappers
├── solver/
│   ├── registry.py           # REGISTRY dict: 19 type codes
│   ├── ocr.py                # Char OCR (ddddocr + CNN fallback)
│   ├── crnn_ocr.py           # CRNN sequential OCR
│   ├── crnn63_ocr.py         # Case-sensitive CRNN63 (main model)
│   ├── slide.py              # Gap detection (alpha mask + edge)
│   ├── click.py              # Word / icon / pass selection
│   ├── logic.py              # Order / spatial / nine-grid
│   ├── preprocess.py         # Unified image loading
│   ├── utils.py              # Helpers (b64, load, etc)
│   ├── trajectory.py         # Mouse tracking generation
│   ├── master_controller.py  # Multi-engine orchestrator
│   └── engines/              # 9 specialized sub-solvers
├── cnn/                      # ONNX pre-trained models
│   ├── crnn2_ocr.onnx        # 15.6 MB
│   ├── crnn63_ocr.onnx       # 8.4 MB
│   └── crnn63_ocr.json       # Vocab metadata
├── web/index.html            # Web debug UI
├── scripts/                  # Test / dev tools
├── logs/                     # Log output
└── __cache__/                # Local cache (auto-created)
```

## Health Check

```bash
curl http://localhost:15666/
# Expected:
# {"code":0,"message":"ok","data":{"service":"syandaV8","version":"2.2.0","types":19,"status":"running"}}
```

## Troubleshooting

| Problem               | Cause                          | Fix                              |
|-----------------------|--------------------------------|----------------------------------|
| Port already in use   | Another process on 15666       | Edit `.env`: `PORT=15667`        |
| Model load failure    | Missing or corrupt ONNX files  | Verify `cnn/` has both .onnx files |
| High memory usage     | Both ONNX models loaded        | Normal (~500 MB)                 |
| TypeError unsupported | Wrong image format             | Use base64, path, or bytes       |
"""

(root / "DEPLOY.md").write_text(deploy_md, encoding="utf-8")
print(f"DEPLOY.md written: {len(deploy_md)} bytes")

# ============ DEPLOY_AI.md ============
deploy_ai = """\
# syandaV8 - AI-Readable Deployment Guide

> Machine-parseable reference for AI agents.
> All paths relative to project root `syandaV8/`.
> No external path dependencies. Single folder is self-contained.

## PROJECT_METADATA

```yaml
name: syandaV8
version: "2.2.0"
python_min: "3.10"
python_tested: "3.14.6"
default_host: "0.0.0.0"
default_port: 15666
onnx_models:
  - path: cnn/crnn2_ocr.onnx
    size_mb: 15.6
  - path: cnn/crnn63_ocr.onnx
    size_mb: 8.4
entrypoint: run.py
api_framework: fastapi
server: uvicorn
total_types: 19
ram_requirement_gb: 2
disk_requirement_mb: 500
```

## TYPE_REGISTRY

```yaml
# 19 captcha type codes and their solve function signatures
types:
  - code: 1001
    name: "mixed english+numbers"
    category: ocr
    fn: solve_ocr63
    params: [img]
  - code: 1002
    name: "pure digits"
    category: ocr
    fn: solve_ocr63
    params: [img]
  - code: 1003
    name: "pure letters"
    category: ocr
    fn: solve_ocr63
    params: [img]
  - code: 1004
    name: "geetest v3 slide"
    category: slide
    fn: _handle_slide
    params: [bg, gap]
  - code: 1005
    name: "geetest v3 click word"
    category: click
    fn: solve_click_word
    params: [img, words]
  - code: 1006
    name: "geetest v3 click icon"
    category: click
    fn: solve_click_icon
    params: [img, icons]
  - code: 1007
    name: "geetest v3 word order"
    category: logic
    fn: solve_word_order
    params: [img, phrase]
  - code: 1008
    name: "geetest v3 spatial reasoning"
    category: logic
    fn: solve_spatial_reasoning
    params: [img, question]
  - code: 1010
    name: "geetest v2/v3 universal"
    category: slide
    fn: _handle_slide
    params: [bg, gap]
  - code: 1012
    name: "geetest v4 slide"
    category: slide
    fn: _handle_slide
    params: [bg, gap]
  - code: 1015
    name: "geetest v4 select hanzi"
    category: click
    fn: solve_click_word
    params: [img, words]
  - code: 1016
    name: "geetest v4 click pass"
    category: click
    fn: solve_click_pass
    params: [img, count]
  - code: 1017
    name: "geetest v4 click icon"
    category: click
    fn: solve_click_icon
    params: [img, icons]
  - code: 1018
    name: "geetest v4 nine grid"
    category: logic
    fn: solve_nine_grid
    params: [img, pos]
  - code: 1019
    name: "geetest v3 nine grid"
    category: logic
    fn: solve_nine_grid
    params: [img, pos]
  - code: 1020
    name: "yidun slide puzzle"
    category: slide
    fn: _handle_slide
    params: [bg, gap]
  - code: 1021
    name: "yidun seamless click"
    category: click
    fn: solve_click_pass
    params: [img, count]
  - code: 1022
    name: "yidun click word"
    category: click
    fn: solve_click_word
    params: [img, words]
  - code: 1023
    name: "yidun click icon"
    category: click
    fn: solve_click_icon
    params: [img, icons]
```

## DEPLOY_STEPS

```yaml
steps:
  - id: 1
    action: verify_python
    command: "python --version"
    expected: "Python 3.10 or later"

  - id: 2
    action: install_deps
    command: "pip install -r requirements.txt"
    expected: "All 11 packages installed successfully"

  - id: 3
    action: create_env
    command: "cp .env.example .env"
    expected: ".env file exists"

  - id: 4
    action: verify_models
    command: "test -f cnn/crnn2_ocr.onnx && test -f cnn/crnn63_ocr.onnx"
    expected: "Both ONNX model files present"

  - id: 5
    action: start_service
    command: "python run.py"
    expected: "Uvicorn running on http://0.0.0.0:15666"

  - id: 6
    action: health_check
    command: "curl -s http://localhost:15666/"
    expected_json:
      code: 0
      data:
        service: syandaV8
        version: "2.2.0"
        types: 19
        status: running

  - id: 7
    action: list_types
    command: "curl -s http://localhost:15666/types"
    expected: "JSON array of 19 type objects with code and name"
```

## API_SPEC

```yaml
base_url: "http://<host>:15666"

endpoints:
  - path: /
    method: GET
    description: "Health check - returns service info and status"
    response:
      code: 0
      data:
        service: "syandaV8"
        version: "2.2.0"
        types: 19
        status: "running"

  - path: /types
    method: GET
    description: "List all registered captcha types"
    response:
      code: 0
      data:
        types:
          - code: 1001
            name: "mixed english+numbers"

  - path: /solve
    method: POST
    content_type: "application/json"
    request:
      type: 1001              # int, required - captcha type code
      image: "<base64>"       # string, required - base64 encoded image
      gap_image: "<base64>"   # string, optional - for slide types
      words: ["A", "B"]       # array, optional - for click_word types
      icons: ["<b64>"]        # array, optional - for click_icon types
      phrase: "HELLO"         # string, optional - for word_order
      question: "left?"       # string, optional - for spatial
      pos: [1, 4, 7]          # array, optional - for nine_grid
      count: 3                # int, optional - for click_pass
    response:
      code: 0
      message: "ok"
      type: 1001
      name: "mixed english+numbers"
      data:
        text: "A7B3"
        confidence: 0.95
      conf: 0.95
      cost_ms: 35.2

  - path: /solve/retry
    method: POST
    description: "Retry solve up to N times until success"
    params:
      attempts: 3             # int, query param, default 3

  - path: /batch_solve
    method: POST
    description: "Batch solve up to 50 images in one request"

  - path: /metrics
    method: GET
    description: "Performance metrics: total requests, by_type stats"
```

## FILE_MANIFEST

```yaml
# Critical files - ALL required for standalone operation
required:
  - path: run.py
    role: "Entry point"
  - path: config.py
    role: "Configuration"
  - path: requirements.txt
    role: "Dependencies"
  - path: api/server.py
    role: "FastAPI server"
  - path: api/response.py
    role: "Response helpers"
  - path: solver/registry.py
    role: "Type router (19 types)"
  - path: solver/preprocess.py
    role: "Image loading pipeline"
  - path: solver/ocr.py
    role: "OCR engine"
  - path: solver/crnn63_ocr.py
    role: "CRNN63 OCR"
  - path: solver/slide.py
    role: "Slide gap detection"
  - path: solver/click.py
    role: "Click word/icon/pass"
  - path: solver/logic.py
    role: "Logic reasoning"
  - path: solver/trajectory.py
    role: "Mouse tracking"
  - path: solver/utils.py
    role: "Utility functions"
  - path: solver/engines/
    role: "9 sub-solvers"
  - path: cnn/crnn2_ocr.onnx
    role: "ONNX model (15.6MB)"
  - path: cnn/crnn63_ocr.onnx
    role: "ONNX model (8.4MB)"
  - path: cnn/crnn63_ocr.json
    role: "Model metadata"

# Optional
optional:
  - path: .env.example
    role: "Config template"
  - path: start.bat
    role: "Windows launcher"
  - path: Dockerfile
    role: "Docker build"
  - path: web/index.html
    role: "Debug UI"
  - path: scripts/
    role: "Dev tools (66 files)"
"""

(root / "DEPLOY_AI.md").write_text(deploy_ai, encoding="utf-8")
print(f"DEPLOY_AI.md written: {len(deploy_ai)} bytes")

# Clean up
import os
os.unlink(__file__)
print("gen_docs.py self-deleted")
