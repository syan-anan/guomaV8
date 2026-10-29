# D5 发布说明

发布日期：2026-10-29
版本号：D5（上一正式版 D4，2026-08-28 部署）

## 本版包含

- 统一验证码入口 `tools/rc_solve_auto.py`
  - 3×3 图像最多提交 2 次（模型正例，上限 4 块）
  - 失败/16 格/词表外类别自动切音频（faster-whisper tiny.en）
  - blocked / noform / noaudio 自动换住宅代理重试（最多 3 次）
- reCAPTCHA 音频求解器 `tools/rc_audio_solve.py`
- Turnstile 测量脚本 `_bk_tmp/_ts_body.py`（本机 IP + chrome_prof）
- 图像线模型与台账（`cnn/rc_tile/`、`reports/d643_ledger.jsonl`）

## 验收

| 项 | 结果 | 证据 |
|---|---|---|
| 通用 OCR 单图 | 91.6–96.0%（六集） | D6.61-C334 全表 |
| Turnstile 真实站点 | 7/7（页面送达时 100%） | reports/ts_rate_home_body.jsonl |
| reCAPTCHA 统一入口 | 28/30 = 93.3%（image 3 / audio 25） | reports/rc_auto_b1.log、rc_auto_b2.log |
| 音频兜底 | 19/20 = 95% | reports/rc_audio_a*.jsonl |
| reCAPTCHA 图像九宫格 | 单题 11.2%，求解 31.5%（未达标） | reports/rc_probe_v20..v35 |

## 已知边界

- 图像九宫格是短板，D5 靠音频兜底通过；单格 precision 0.90，目标需要 ~0.98。
- 音频通过依赖会话可信度（转写内容经常语义不通仍出 token），Google 收紧会掉。
- reCAPTCHA 必须住宅代理；Turnstile 必须本机 IP + 真实 profile，两条线不可混用。
- 站点 429 会阻断 Turnstile 连续快跑，需间隔重试。

## 升级说明

- 内部基准编号 D6.61-C334 保留在 reports 中作为历史标签，对外版本号统一为 D5。
- 上一版 D4 的 VERSION/README 备份见 `VERSION.bak_*`、`README.md.bak_*`。
