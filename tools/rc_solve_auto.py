# -*- coding: utf-8 -*-
"""rc_solve_auto: unified reCAPTCHA entry.

Policy:
1) open demo with a fresh residential proxy
2) if the served challenge is 3x3 and the category is known, spend at most 2 image submissions
   (model positives, cap 4 tiles); any token -> done
3) otherwise/after failure switch to the audio challenge and solve with faster-whisper
4) blocked ("automated queries") -> rotate proxy

Output per attempt: mode that produced the token, submits used, timing.
"""
import json, pathlib, base64, re, sys, time, urllib.request
import numpy as np, cv2, onnxruntime as ort, av
from patchright.sync_api import sync_playwright
from faster_whisper import WhisperModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
C = ROOT/"_bk_tmp"/"clipl"
TMP = ROOT/"_bk_tmp"/"rc_audio"; TMP.mkdir(parents=True, exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TAG = sys.argv[2] if len(sys.argv) > 2 else "auto"
URL = sys.argv[3] if len(sys.argv) > 3 else "https://www.recaptcha.net/recaptcha/api2/demo"
CATS = ["摩托车","自行车","公交车","消防栓","红绿灯","汽车","小轿车","人行横道","楼梯","船","棕榈树","烟囱","出租车"]
MEAN = np.array([0.48145466,0.4578275,0.40821073], np.float32)
STD = np.array([0.26862954,0.26130258,0.27577711], np.float32)
W2D = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","oh":"0","to":"2","too":"2","for":"4","ate":"8","won":"1"}

vis = ort.InferenceSession(str(C/"onnx_vision_model_quantized.onnx"), providers=["CPUExecutionProvider"])
head_sess = ort.InferenceSession(str(ROOT/"cnn"/"rc_tile"/"rc_head_da.onnx"), providers=["CPUExecutionProvider"])
mdl = WhisperModel(str(ROOT/"_bk_tmp"/"whisper_tiny_en"), device="cpu", compute_type="int8")

GRID = """()=>{const els=[...document.querySelectorAll('td')].filter(e=>{const r=e.getBoundingClientRect(); return r.width>40&&r.height>40&&r.width<500;});
 const uniq=(v)=>{const s=[]; v.sort((a,b)=>a-b).forEach(x=>{if(!s.length||Math.abs(x-s[s.length-1])>6) s.push(x);}); return s;};
 const xs=uniq(els.map(e=>Math.round(e.getBoundingClientRect().x))); const ys=uniq(els.map(e=>Math.round(e.getBoundingClientRect().y)));
 if(xs.length<3||ys.length<3||xs.length>6||ys.length>6) return {cells:[]};
 const o=[]; for(const y of ys) for(const x of xs){const c=els.filter(e=>{const r=e.getBoundingClientRect(); return Math.abs(r.x-x)<4&&Math.abs(r.y-y)<4;}).sort((a,b)=>b.getBoundingClientRect().width-a.getBoundingClientRect().width); if(!c.length) return {cells:[]}; const r=c[0].getBoundingClientRect(); o.push({id:c[0].id,x:r.x,y:r.y,w:r.width,h:r.height});}
 return {nx:xs.length,ny:ys.length,cells:o};}"""
PROMPT = "()=>{const d=document.querySelector('.rc-imageselect-desc-wrapper'); return (d&&d.innerText)||'';}"
TOK = "()=>{const e=document.querySelector('[name=g-recaptcha-response]'); return e?e.value.length:0;}"
MSGS = """()=>{const out=[];document.querySelectorAll('div,span').forEach(e=>{const c=String(e.className||''); if(c.includes('error')||c.includes('incorrect')){const s=getComputedStyle(e);const r=e.getBoundingClientRect(); if(s.display!=='none'&&r.width>1&&r.height>1) out.push(c);}}); return out;}"""
BODY = "()=>document.body.innerText.slice(0,120).replace(/\\n/g,' ')"
ERR = "()=>{const e=document.querySelector('.rc-audiochallenge-error-message'); return e?(e.innerText||'').trim().slice(0,60):''}"
AUDIO = """async ()=>{const a=document.querySelector('audio'); if(!a||!a.src) return ''; try{const r=await fetch(a.src,{credentials:'include'}); const b=await r.blob(); return await new Promise(res=>{const fr=new FileReader(); fr.onload=()=>res(String(fr.result).split(',')[1]); fr.readAsDataURL(b);});}catch(e){return ''}}"""
API = "https://service.ipzan.com/core-extract?num=1&no=20260205199812029445&minute=10&pool=quality&secret=acng2lp2m7250io"

def newproxy(minute=10):
    try:
        raw = urllib.request.urlopen(urllib.request.Request(API, headers={"User-Agent":"Mozilla/5.0"}), timeout=25).read().decode("utf-8","ignore")
        m = re.search(r'(\d{1,3}(?:\.\d{1,3}){3}):(\d{2,5})', raw)
        return m.group(1)+":"+m.group(2) if m else None
    except Exception:
        return None

def emb(im):
    im = cv2.cvtColor(cv2.resize(im,(224,224),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((im.astype(np.float32)/255.0-MEAN)/STD).transpose(2,0,1)[None]
    return vis.run(None, {"pixel_values":x})[0][0]

def sc(im, cat):
    oh = np.zeros(len(CATS), np.float32); oh[CATS.index(cat)] = 1
    x = np.concatenate([emb(im), oh])[None].astype(np.float32)
    logit = float(head_sess.run(None, {"x": x})[0][0][0])
    return float(1.0/(1.0+np.exp(-logit)))

def shot(p, ib, t):
    try:
        raw = p.screenshot(clip={"x":ib["x"]+t["x"],"y":ib["y"]+t["y"],"width":t["w"],"height":t["h"]})
        return cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def load_audio(path):
    with av.open(path) as c:
        st = c.streams.audio[0]; res = av.AudioResampler(format="fltp",layout="mono",rate=16000); out=[]
        for fr in c.decode(st):
            for f in res.resample(fr): out.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None): out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)

def norm(t):
    toks = [W2D.get(w,w) for w in re.split(r"\s+", t.lower().replace(","," ")) if w]
    return re.sub(r"[^a-z0-9]","", "".join(toks))

def image_attempt(p, bf, rec):
    """At most 2 submissions. Returns True if token issued."""
    for sub in range(2):
        g = bf.evaluate(GRID); prompt = bf.evaluate(PROMPT)
        cat = next((c for c in CATS if c in prompt), None)
        if len(g["cells"]) != 9 or not cat:
            rec["image"] = "skip-grid-%s-%s" % (len(g["cells"]), cat); return False
        ib = p.query_selector("iframe[src*='bframe']").bounding_box()
        probs = []
        for t in g["cells"]:
            im = shot(p, ib, t); probs.append(round(sc(im,cat),3) if im is not None else 0.0)
        order = [int(i) for i in np.argsort(-np.array(probs))]
        pick = [i for i in order if probs[i] >= 0.8][:4] or order[:3]
        for i in pick:
            t = g["cells"][i]
            p.mouse.click(ib["x"]+t["x"]+t["w"]/2, ib["y"]+t["y"]+t["h"]/2); p.wait_for_timeout(4800)
        try: bf.locator("#recaptcha-verify-button").click(timeout=4000)
        except Exception: pass
        p.wait_for_timeout(5200)
        msgs = bf.evaluate(MSGS); tok = p.evaluate(TOK)
        rec.setdefault("image_subs", []).append({"cat":cat,"pick":pick,"msgs":msgs,"tok":tok})
        if tok > 20:
            rec["mode"] = "image"; return True
        if "incorrect" in " ".join(msgs):
            return False
    return False

def audio_attempt(p, bf, rec, tag):
    entered = False
    for _ in range(6):
        if bf.query_selector("#audio-response"):
            entered = True; break
        try: bf.locator("#recaptcha-audio-button").click(timeout=3500)
        except Exception: pass
        p.wait_for_timeout(2600)
        body = bf.evaluate(BODY)
        if "稍后重试" in body or "自动查询" in body:
            rec["audio_error"] = "blocked"; return False, "blocked"
        if bf.query_selector("#audio-response"):
            entered = True; break
    if not entered:
        rec["audio_error"] = "noform"; return False, "noform"
    for r in range(4):
        b64 = ""
        for _ in range(20):
            b64 = bf.evaluate(AUDIO)
            if b64: break
            p.wait_for_timeout(700)
        if not b64:
            rec["audio_error"] = "noaudio"; return False, "noaudio"
        fp = TMP/("%s_%d_%d.mp3" % (tag, int(time.time()) % 100000, r))
        fp.write_bytes(base64.b64decode(b64))
        segs, _ = mdl.transcribe(load_audio(str(fp)), language="en", beam_size=5)
        txt = " ".join(s.text.strip() for s in segs).strip(); clean = norm(txt)
        bf.fill("#audio-response", clean); p.wait_for_timeout(500)
        try: bf.locator("#recaptcha-verify-button").click(timeout=5000)
        except Exception: pass
        p.wait_for_timeout(5000)
        tok = p.evaluate(TOK); err = bf.evaluate(ERR)
        rec.setdefault("audio_rounds", []).append({"raw":txt[:50],"clean":clean,"tok":tok,"err":err})
        if tok > 20:
            rec["mode"] = "audio"; return True, None
        if ("更多" in err) or ("more" in err.lower()):
            try: bf.locator("#recaptcha-reload-button").click(timeout=4000)
            except Exception: pass
            p.wait_for_timeout(3200); continue
        break
    return False, None

def one(pw, k):
    px = newproxy(10)
    args = ["--disable-blink-features=AutomationControlled","--window-position=-32000,-32000","--window-size=1290,900"]
    if px: args.append("--proxy-server=http://"+px)
    ctx = pw.chromium.launch_persistent_context(user_data_dir=str(ROOT/"_bk_tmp"/"rc_prof_px2"), channel="chrome",
        headless=False, viewport={"width":1280,"height":860}, args=args)
    rec = {"k":k,"proxy":px,"t0":time.time()}
    try:
        p = ctx.pages[0] if ctx.pages else ctx.new_page()
        p.goto(URL, wait_until="domcontentloaded", timeout=45000); p.wait_for_timeout(2000)
        fr = [f for f in p.frames if "/recaptcha/api2/anchor" in (f.url or "")]
        if fr:
            bb = fr[0].locator("#recaptcha-anchor").bounding_box()
            if bb: p.mouse.click(bb["x"]+14, bb["y"]+14)
        bf = None
        for _ in range(14):
            p.wait_for_timeout(700); bf = next((f for f in p.frames if "/recaptcha/api2/bframe" in (f.url or "")), None)
            if bf: break
        if not bf:
            rec["err"] = "nobf"; return rec
        p.wait_for_timeout(2400)
        if p.evaluate(TOK) > 20:
            rec.update({"passed":True,"mode":"auto"}); return rec
        if image_attempt(p, bf, rec):
            rec["passed"] = True; return rec
        ok, why = audio_attempt(p, bf, rec, "auto%d" % k)
        rec["passed"] = bool(ok); rec["audio_block"] = why
        return rec
    except Exception as e:
        rec["err"] = str(e)[:80]; return rec
    finally:
        rec["secs"] = round(time.time()-rec["t0"],1)
        try: ctx.close()
        except Exception: pass

ok = 0; n = 0; modes = {"image":0,"audio":0,"auto":0,None:0}
with sync_playwright() as pw:
    for k in range(N):
        rec = None
        for retry in range(3):
            rec = one(pw, k)
            if rec.get("passed"): break
            if rec.get("audio_block") in ("blocked","noform","noaudio") or rec.get("err"):
                continue
            break
        n += 1
        passed = bool(rec and rec.get("passed")); ok += passed
        modes[rec.get("mode") if rec else None] = modes.get(rec.get("mode") if rec else None, 0) + (1 if passed else 0)
        out = {"k":k,"passed":passed,"rate":[ok,n],"mode":rec.get("mode") if rec else None,"res":rec}
        print(json.dumps(out, ensure_ascii=False), flush=True)
        with open(ROOT/"reports"/("rc_auto_%s.jsonl" % TAG), "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
print("RATE", ok, "/", n, "MODES", modes, flush=True)
