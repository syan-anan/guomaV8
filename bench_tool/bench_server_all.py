# -*- coding: utf-8 -*-
"""Benchmark every server type against known-answer data.
Datasets: behavior_500 (slide/word/icon/order/spatial/pass/grid) + bank1500_real (OCR).
Types without datasets (1009/1011/1014/1024/1025/1026) are exercised with synthetic known-answer images.
"""
import argparse, base64, io, json, math, pathlib, random, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(r"H:\qinglong\ddddocr")
BEH = ROOT/"xunlian"/"validate"/"behavior_500"
BANK = ROOT/"xunlian"/"validate"/"testset_200"
SERVER = "http://160.202.254.65:15666"

def b64p(p): return base64.b64encode(pathlib.Path(p).read_bytes()).decode()
def b64img(im):
    b=io.BytesIO(); im.save(b,format="PNG"); return base64.b64encode(b.getvalue()).decode()
def post(t,image,gap=None,extra=None,timeout=40):
    d={"type":t,"image":image}
    if gap: d["gap_image"]=gap
    if extra: d["extra"]=extra
    req=urllib.request.Request(SERVER+"/solve",data=json.dumps(d).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    r=json.loads(urllib.request.urlopen(req,timeout=timeout).read().decode())
    r["_ms"]=round((time.time()-t0)*1000,1)
    time.sleep(0.55)
    return r
def unwrap(d):
    while isinstance(d,dict) and isinstance(d.get("data"),dict) and not any(k in d for k in ("x","y","points","angle","distance","total_distance","trajectory")):
        d=d["data"]
    return d if isinstance(d,dict) else {}
def pts(d):
    d=unwrap(d)
    out=[]
    for k in ("points","trajectory"):
        v=d.get(k)
        if isinstance(v,list):
            for p in v:
                if isinstance(p,dict) and "x" in p and "y" in p: out.append((float(p["x"]),float(p["y"])))
    if "x" in d and "y" in d: out.append((float(d["x"]),float(d["y"])))
    return out
def hit_all(d,pos,tol=25):
    got=pts(d)
    return bool(got) and all(any(abs(gx-x)<=tol and abs(gy-y)<=tol for gx,gy in got) for x,y in pos)

def synth_rotation(angle):
    im=Image.new("RGB",(200,200),"white"); dr=ImageDraw.Draw(im)
    box=Image.new("RGB",(120,60),"white"); db=ImageDraw.Draw(box); db.rectangle([10,10,110,50],fill="black")
    box=box.rotate(angle,expand=False,fillcolor="white")
    im.paste(box,(40,70)); return im
def synth_match3():
    im=Image.new("RGB",(220,220),"white"); dr=ImageDraw.Draw(im)
    cx,cy=110,110; dr.rectangle([cx-45,cy-45,cx+45,cy+45],fill=(220,30,30))
    dr.rectangle([20,20,50,50],fill=(30,180,30)); dr.rectangle([170,20,200,50],fill=(30,80,220))
    return im,(cx,cy)
def synth_drawing():
    im=Image.new("RGB",(220,220),"white"); dr=ImageDraw.Draw(im)
    dr.line([(30,60),(190,160)],fill="black",width=6); return im,(110,110)
def synth_calendar():
    im=Image.new("RGB",(240,200),"white"); dr=ImageDraw.Draw(im)
    for r in range(4):
        for c in range(5): dr.rectangle([10+c*44,10+r*44,50+c*44,50+r*44],outline=(180,180,180))
    dr.rectangle([54,54,94,94],fill=(40,90,220)); return im,(74,74)
def synth_combo():
    im=Image.new("RGB",(220,120),"white"); dr=ImageDraw.Draw(im); pos=[]
    for i,(x,y) in enumerate([(30,40),(90,40),(150,40)]):
        dr.rectangle([x-12,y-12,x+12,y+12],fill="black"); pos.append((x,y))
    return im,pos

CALIB={}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=50); ap.add_argument("--calib",action="store_true"); ap.add_argument("--types",default="")
    a=ap.parse_args()
    beh=json.loads((BEH/"labels.json").read_text(encoding="utf-8"))
    bank=json.loads((BANK/"labels.json").read_text(encoding="utf-8"))
    rng=random.Random(4080)
    tasks=[]
    def add(t,items):
        for it in items[:a.n]: tasks.append((t,it))
    for t,fam in [(1004,"slide"),(1010,"slide"),(1012,"slide"),(1020,"slide"),(1011,"slide")]: add(t,beh[fam])
    for t,fam in [(1005,"word"),(1015,"word"),(1022,"word")]: add(t,beh[fam])
    for t,fam in [(1006,"icon"),(1017,"icon"),(1023,"icon")]: add(t,beh[fam])
    add(1007,beh["order"]); add(1008,beh["spatial"])
    for t in (1016,1021): add(t,beh["pass"])
    for t in (1018,1019): add(t,beh["grid"])
    for t in (1001,1002,1003):
        items=[b for b in bank if b["type"]==t]; rng.shuffle(items); add(t,items)
    if a.types:
        keep={int(x) for x in a.types.split(",")}; tasks=[x for x in tasks if x[0] in keep]
    res={}; lat={}
    total=len(tasks)+6*(3 if a.calib else a.n)
    i=0
    for t,it in tasks:
        i+=1
        fam=None
        if t in (1004,1010,1012,1020,1011): fam="slide"
        elif t in (1005,1015,1022): fam="word"
        elif t in (1006,1017,1023): fam="icon"
        elif t==1007: fam="order"
        elif t==1008: fam="spatial"
        elif t in (1016,1021): fam="pass"
        elif t in (1018,1019): fam="grid"
        try:
            if fam=="slide":
                r=post(t,b64p(BEH/it["bg"]),gap=b64p(BEH/it["gap"]))
                d=unwrap(r.get("data",{})); val=d.get("distance") if t!=1011 else d.get("total_distance")
                ok=(r.get("code")==0 and val is not None and abs(float(val)-float(it["distance"]))<=12)
            elif fam=="word":
                r=post(t,b64p(BEH/it["img"]),extra={"words":it["words"]}); ok=(r.get("code")==0 and hit_all(r.get("data",{}),it["positions"]))
            elif fam=="icon":
                r=post(t,b64p(BEH/it["img"]),extra={"icons":it["icons"]}); ok=(r.get("code")==0 and hit_all(r.get("data",{}),it["positions"]))
            elif fam=="order":
                r=post(t,b64p(BEH/it["img"]),extra={"phrase":it["phrase"]}); ok=(r.get("code")==0 and hit_all(r.get("data",{}),it["positions"]))
            elif fam=="spatial":
                r=post(t,b64p(BEH/it["img"])); ok=(r.get("code")==0 and hit_all(r.get("data",{}),it["positions"]))
            elif fam=="pass":
                r=post(t,b64p(BEH/it["img"]),extra={"count":it["count"]}); ok=(r.get("code")==0 and len(pts(r.get("data",{})))>=int(it["count"]))
            elif fam=="grid":
                r=post(t,b64p(BEH/it["img"]),extra={"positions":it["positions"]}); d=r.get("data",{})
                coords=[it["cells"][str(k)] for k in it["positions"] if str(k) in it["cells"]]
                ok=(r.get("code")==0 and len(pts(d))==len(coords) and hit_all(d,coords))
            else:
                r=post(t,b64p(BANK/it["file"])); ok=(r.get("code")==0 and str(r.get("data",{}).get("text","")).upper().replace(" ","")==it["label"].upper())
        except Exception as e:
            r={"code":-99,"message":str(e)[:80]}; ok=False
        s=res.setdefault(t,[0,0]); s[1]+=1; s[0]+=1 if ok else 0
        lat.setdefault(t,[]).append(r.get("_ms",0))
        if i%50==0: print(f"progress {i}/{total}",flush=True)
    # synthetic types
    for t in (1009,1011,1014,1024,1025,1026):
        if a.types and t not in {int(x) for x in a.types.split(",")}: continue
        n=3 if a.calib else a.n
        for k in range(n):
            try:
                if t==1009:
                    ang=rng.uniform(-40,40); im=synth_rotation(ang); truth=ang
                    r=post(t,b64img(im)); got=float(unwrap(r.get("data",{})).get("angle",0)); ok=(r.get("code")==0 and abs(got-truth)<=12)
                    extra=(round(truth,1),got)
                elif t==1011:
                    it=beh["slide"][k%len(beh["slide"])]; r=post(t,b64p(BEH/it["bg"]),gap=b64p(BEH/it["gap"])); got=unwrap(r.get("data",{})).get("total_distance")
                    ok=(r.get("code")==0 and got is not None and abs(float(got)-float(it["distance"]))<=12); extra=(it["distance"],got)
                elif t==1014:
                    im,c=synth_match3(); r=post(t,b64img(im)); got=pts(r.get("data",{})); ok=(r.get("code")==0 and bool(got) and abs(got[0][0]-c[0])<=35 and abs(got[0][1]-c[1])<=35); extra=(c,got[:1])
                elif t==1024:
                    im,c=synth_drawing(); r=post(t,b64img(im)); got=pts(r.get("data",{})); ok=(r.get("code")==0 and bool(got) and abs(got[0][0]-c[0])<=45 and abs(got[0][1]-c[1])<=45); extra=(c,got[:1])
                elif t==1025:
                    im,c=synth_calendar(); r=post(t,b64img(im)); got=pts(r.get("data",{})); ok=(r.get("code")==0 and bool(got) and abs(got[0][0]-c[0])<=30 and abs(got[0][1]-c[1])<=30); extra=(c,got[:1])
                else:
                    im,pos=synth_combo(); r=post(t,b64img(im)); d=r.get("data",{}); ok=(r.get("code")==0 and hit_all(d,pos,30)); extra=(pos,pts(d)[:4])
            except Exception as e:
                r={"code":-99,"message":str(e)[:80]}; ok=False; extra=None
            if a.calib: print("CALIB",t,ok,extra,r.get("message","")[:60],flush=True)
            s=res.setdefault(t,[0,0]); s[1]+=1; s[0]+=1 if ok else 0
            lat.setdefault(t,[]).append(r.get("_ms",0))
    print("\nTYPE  pass/total  rate   avg_ms")
    for t in sorted(res):
        p,n=res[t]; ms=lat.get(t,[]); avg=round(sum(ms)/len(ms),1) if ms else 0
        print(f"{t:<6}{p:>3}/{n:<5}{100*p/max(1,n):>6.1f}%  {avg}")
    out=ROOT/"reports"/"server_all_types_bench.json"
    out.write_text(json.dumps({"server":SERVER,"n":a.n,"ts":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results":{str(t):{"pass":v[0],"total":v[1],"rate":round(v[0]/max(1,v[1]),4),
        "avg_ms":round(sum(lat.get(t,[0]))/max(1,len(lat.get(t,[]))),1)} for t,v in res.items()}},ensure_ascii=False,indent=1),encoding="utf-8")
    print("saved",out)

if __name__=="__main__": main()
