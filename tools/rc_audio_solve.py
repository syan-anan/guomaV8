# -*- coding: utf-8 -*-
"""Audio solver v4: rotate proxy on the 'automated queries' block; one browser per attempt."""
import json, pathlib, base64, re, sys, urllib.request, time, numpy as np, av
from patchright.sync_api import sync_playwright
from faster_whisper import WhisperModel
ROOT=pathlib.Path(__file__).resolve().parent.parent; TMP=ROOT/"_bk_tmp"/"rc_audio"; TMP.mkdir(parents=True,exist_ok=True)
MODEL=str(ROOT/"_bk_tmp"/"whisper_tiny_en")
N=int(sys.argv[1]) if len(sys.argv)>1 else 10
TAG=sys.argv[2] if len(sys.argv)>2 else "a5"
W2D={"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","oh":"0","to":"2","too":"2","for":"4","ate":"8","won":"1"}
mdl=WhisperModel(MODEL,device="cpu",compute_type="int8")
def load_audio(path):
    with av.open(path) as c:
        st=c.streams.audio[0]; res=av.AudioResampler(format="fltp",layout="mono",rate=16000); out=[]
        for fr in c.decode(st):
            for f in res.resample(fr): out.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None): out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)
def norm(t):
    toks=[W2D.get(w,w) for w in re.split(r"\s+",t.lower().replace(","," ")) if w]
    return re.sub(r"[^a-z0-9]","","".join(toks))
AUDIO="""async ()=>{const a=document.querySelector('audio'); if(!a||!a.src) return ''; try{const r=await fetch(a.src,{credentials:'include'}); const b=await r.blob(); return await new Promise(res=>{const fr=new FileReader(); fr.onload=()=>res(String(fr.result).split(',')[1]); fr.readAsDataURL(b);});}catch(e){return ''}}"""
TOK="()=>{const e=document.querySelector('[name=g-recaptcha-response]'); return e?e.value.length:0;}"
ERR="()=>{const e=document.querySelector('.rc-audiochallenge-error-message'); return e?(e.innerText||'').trim().slice(0,60):''}"
BODY="()=>document.body.innerText.slice(0,120).replace(/\\n/g,' ')"
def newproxy(minute=10):
    u=f"https://service.ipzan.com/core-extract?num=1&no=20260205199812029445&minute={minute}&pool=quality&secret=acng2lp2m7250io"
    raw=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"}),timeout=25).read().decode("utf-8","ignore")
    m=re.search(r'(\d{1,3}(?:\.\d{1,3}){3}):(\d{2,5})',raw); return (m.group(1)+":"+m.group(2)) if m else None
def one_attempt(pw, k, ip_try):
    px=newproxy(10)
    args=["--disable-blink-features=AutomationControlled","--window-position=-32000,-32000","--window-size=1290,900"]
    if px: args.append("--proxy-server=http://"+px)
    ctx=pw.chromium.launch_persistent_context(user_data_dir=str(ROOT/"_bk_tmp"/"rc_prof_px2"),channel="chrome",headless=False,
        viewport={"width":1280,"height":860},args=args)
    out={"ip_try":ip_try,"proxy":px}
    try:
        p=ctx.pages[0] if ctx.pages else ctx.new_page()
        p.goto("https://www.recaptcha.net/recaptcha/api2/demo",wait_until="domcontentloaded",timeout=45000); p.wait_for_timeout(2000)
        fr=[f for f in p.frames if "/recaptcha/api2/anchor" in (f.url or "")]
        if fr:
            bb=fr[0].locator("#recaptcha-anchor").bounding_box()
            if bb: p.mouse.click(bb["x"]+14,bb["y"]+14)
        bf=None
        for _ in range(14):
            p.wait_for_timeout(700); bf=next((f for f in p.frames if "/recaptcha/api2/bframe" in (f.url or "")),None)
            if bf: break
        if not bf: out["err"]="nobf"; return out
        p.wait_for_timeout(2400)
        entered=False
        for _ in range(6):
            if bf.query_selector("#audio-response"): entered=True; break
            try: bf.locator("#recaptcha-audio-button").click(timeout=3500)
            except Exception: pass
            p.wait_for_timeout(2600)
            body=bf.evaluate(BODY)
            if "稍后重试" in body or "自动查询" in body:
                out["err"]="blocked"; out["body"]=body[:60]; return out
            if bf.query_selector("#audio-response"): entered=True; break
        if not entered: out["err"]="noform"; out["body"]=bf.evaluate(BODY)[:60]; return out
        rounds=[]
        for r in range(4):
            b64=""
            for _ in range(20):
                b64=bf.evaluate(AUDIO)
                if b64: break
                p.wait_for_timeout(700)
            if not b64: rounds.append({"err":"noaudio"}); break
            fp=TMP/("v%d_%d_%d.mp3"%(k,ip_try,r)); fp.write_bytes(base64.b64decode(b64))
            segs,_=mdl.transcribe(load_audio(str(fp)),language="en",beam_size=5)
            txt=" ".join(s.text.strip() for s in segs).strip(); clean=norm(txt)
            bf.fill("#audio-response",clean); p.wait_for_timeout(500)
            try: bf.locator("#recaptcha-verify-button").click(timeout=5000)
            except Exception: pass
            p.wait_for_timeout(5000)
            tok=p.evaluate(TOK); err=bf.evaluate(ERR)
            rounds.append({"raw":txt[:50],"clean":clean,"tok":tok,"err":err})
            if tok>20: out.update({"passed":True,"tok":tok,"rounds":rounds}); return out
            if ("更多" in err) or ("more" in err.lower()):
                try: bf.locator("#recaptcha-reload-button").click(timeout=4000)
                except Exception: pass
                p.wait_for_timeout(3200); continue
            break
        out.update({"passed":False,"rounds":rounds})
        return out
    except Exception as e:
        out["err"]=str(e)[:60]; return out
    finally:
        try: ctx.close()
        except Exception: pass
ok=0; tried=0; blocked=0
with sync_playwright() as pw:
    for k in range(N):
        res=None
        for ip_try in range(4):
            res=one_attempt(pw,k,ip_try)
            if res.get("passed") or res.get("err") in (None,):
                break
            if res.get("err") in ("blocked","noform"):
                blocked+=1; continue
            break
        tried+=1
        passed=bool(res and res.get("passed"))
        ok+=passed
        rec={"k":k,"passed":passed,"rate":[ok,tried],"blocked_total":blocked,"res":res}
        print(json.dumps(rec,ensure_ascii=False),flush=True)
        with open(ROOT/"reports"/("rc_audio_%s.jsonl"%TAG),"a",encoding="utf-8") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
print("RATE",ok,"/",tried,flush=True)
