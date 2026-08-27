# AI Deployment Instructions - syandaV8 D3

## Package Summary
| Field | Value |
|-------|-------|
| Name | syandaV8 |
| Version | D3 |
| Root | {workspace}/syandaV8 |
| Total Size | ~39 MB (24 MB models + 15 MB code) |
| File Count | 198 files (non-pycache) |
| Self-contained | Yes - all paths resolved via __file__ |
| Path Dependencies | None |

## Prerequisites
| Requirement | Specification |
|-------------|---------------|
| Language Runtime | Python >= 3.10 |
| Architecture | x86_64 or ARM64 |
| Min RAM | 2 GB |
| Min Disk | 500 MB |
| GPU Required | No (pure CPU inference) |
| Supported OS | Linux, macOS, Windows Server |

## Dependency Matrix
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.141.1 | HTTP server framework |
| uvicorn | 0.52.4 | ASGI server |
| ddddocr | 1.6.1 | Primary OCR engine |
| opencv-python | 5.0.0.93 | Image preprocessing |
| numpy | 2.1.3 | Numerical operations |
| pillow | 12.3.0 | Image format handling |
| onnxruntime | >=1.15.0 | ONNX model inference |
| python-dotenv | 1.0.1 | Env var loading |
| python-multipart | 0.0.32 | Form-data parsing |
| loguru | 0.7.3 | Structured logging |
| httpx | 0.28.1 | HTTP client |
| apscheduler | 3.11.3 | Scheduled tasks |

Install: pip install -r requirements.txt

## Model Artifacts (in cnn/)
| File | Size | Description |
|------|------|-------------|
| crnn2_ocr.onnx | ~15.6 MB | CRNN-2 character recognition |
| crnn63_ocr.onnx | ~8.4 MB | CRNN-63 character recognition |
| crnn63_ocr.json | 230 B | Label vocabulary mapping |

Total models: ~24 MB
Models load at startup via solver/ocr.py::preload_ocr_models() wrapped in try/except.
## Service Configuration
| Env Variable | Default | Description |
|-------------|---------|-------------|
| HOST | 0.0.0.0 | Bind address |
| PORT | 15666 | Listen port |
| OCR_RETRY | 3 | OCR fallback retries |
| ADMIN_KEY | changeme | Admin API key |
| LOG_LEVEL | info | Log level |
| CACHE_LIMIT | 1000 | Max cache entries |
| INFER_TIMEOUT | 30000 | Inference timeout ms |
| RATE_LIMIT_PER_MINUTE | 120 | Rate limit per IP |

Config file: .env (copy from .env.example)

## API Endpoints
Base URL: http://{HOST}:{PORT}

| Method | Path | Input | Output |
|--------|------|-------|--------|
| GET | /health | - | Health check |
| GET | /types | - | Type list 19 types |
| GET | /docs | - | Swagger UI |
| GET | /metrics | - | Performance stats |
| POST | /solve | JSON body | Solution response |
| POST | /solve/retry | JSON+param | Best-of-N solution |
| POST | /batch_solve | JSON array | Array of solutions |
| POST | /apiv1/ocr | form-data | OCR result |
| POST | /apiv1/slide | form-data | Slide solution |

## Request Schema
type: integer (required)
image: string base64 data URI (required)
gap_image: optional string
extra: optional object

## Response Schema
code: integer 0=ok, negative=error
message: string status message
type: integer type code
name: string captcha type name
data: object solution payload
conf: float confidence score 0-1
cost_ms: float processing time in ms
## Startup Sequence
1. run.py sets cwd to script parent directory
2. Sets cache env vars (PIP_CACHE_DIR, HF_HOME, etc.)
3. Imports api.server -> FastAPI app created
4. Startup event fires: loads ONNX models
5. uvicorn listens on HOST:PORT

Cold start: 2-5 seconds (model loading)

## Type Code Registry (machine-readable)
1001=ocr_alphanum  1002=ocr_digits  1003=ocr_letters
1004=geetest3_slider  1005=geetest3_point_char
1006=geetest3_point_obj  1007=geetest3_word_order
1008=geetest3_spatial  1010=geetest23_universal_slider
1012=geetest4_slider  1015=geetest4_cjk_char
1016=geetest4_point_pass  1017=geetest4_icon_select
1018=geetest4_nine_grid  1019=geetest3_nine_grid
1020=yidun_slider  1021=yidun_invisible_point
1022=yidun_point_char  1023=yidun_point_icon

## Automated Deploy Steps
Step 1: cp -r syandaV8/ /opt/syandaV8
Step 2: cd /opt/syandaV8
Step 3: test -f cnn/crnn2_ocr.onnx || exit 1
Step 4: test -f cnn/crnn63_ocr.onnx || exit 1
Step 5: python3 -m venv .venv && source .venv/bin/activate
Step 6: pip install -q -r requirements.txt
Step 7: python run.py &
Step 8: sleep 3 && curl -sf http://localhost:15666/

## Verification Checklist
[ ] Models exist at cnn/ (24 MB total)
[ ] requirements.txt has 12 entries
[ ] run.py runs without ImportError
[ ] Models load at startup (check console output)
[ ] GET / returns status running
[ ] GET /types returns 19 entries
[ ] Port 15666 accessible externally

## Rollback Plan
1. Stop: kill all python run.py processes
2. Restore: mv syandaV8_old syandaV8
3. Restart: cd syandaV8 && python run.py &

## Monitoring
Health: curl -sf http://localhost:15666/health every 10s
Metrics: curl http://localhost:15666/metrics
Logs: tail -f logs/server.log

Auto-deploy instruction set D3 | Updated: 2026-08-27
