# -*- coding: utf-8 -*-
"""One-shot local queue: rebuilt C6 anchors -> PL -> distill -> stage47 bundle/smoke."""
from __future__ import annotations
import gc, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from PIL import Image

REPRO = Path(r"E:\复现"); SOURCE = REPRO / "workspace" / "source"; STRICT = REPRO / "runs" / "strict"
TEACHER = REPRO / "runs" / "segformer_teacher_historical"; LOGS = STRICT / "logs"
PYTHON = Path(r"D:\Anaconda3\python.exe")
STATUS = STRICT / "overnight_segformer_pl_distill_47_status.json"
REPORT = STRICT / "OVERNIGHT_SEGFORMER_PL_DISTILL_47_FINAL_20260926.md"
MASTER_LOG = LOGS / "overnight_segformer_pl_distill_47.log"
BASE = STRICT / "ckpt_kan_from_v4_1_7439_16k.pth"
BASE_SHA = "806039602f21cb6da342085f867727712a9c06f40c8765c031791c07a5882798"
CACHE = STRICT / "pl_cache_kan_from_v4_1_7439_ms2"
C6_PRED = TEACHER / "c6_test1_anchor_candidate"
C6_VALIDATION = TEACHER / "audits" / "c6_test1_validation_repair1.json"
IDENTITY = TEACHER / "audits" / "historical_anchor_identity.json"
C6_CKPT = TEACHER / "outputs" / "segformer_b2_c6_dacs768" / "final.pt"
C6_CKPT_SHA = "028437935a7d4a60ec39b28b8484801a6aaa61561529a1516a13f1a46d212ad5"
ANCHOR_NAME = "teacher_anchors_segformer_historical_rebuilt"; ANCHORS = STRICT / ANCHOR_NAME
ANCHOR_MANIFEST = STRICT / f"{ANCHOR_NAME}_MANIFEST.json"
PL_MAIN = "ckpt_kan_historical_pl_rebuilt.pth"; PL_SNAPS = [f"ckpt_kan_historical_pl_rebuilt_i{i}.pth" for i in (1000,2000,3000,4000)]
PL_EVAL = "eval_kan_historical_pl_rebuilt_i2000_plain_ema_n500.json"; PL_MASKS = STRICT / "pl_masks_historical_rebuilt_t85"
DIST_MAIN = "ckpt_kan_segformer_distill_rebuilt.pth"; DIST_SNAPS = [f"ckpt_kan_segformer_distill_rebuilt_i{i}.pth" for i in (1000,2000,3000,4000)]
DIST_EVAL = "eval_kan_segformer_distill_rebuilt_i2000_plain_ema_n500.json"
CONSISTENCY = "audit_kan_segformer_distill_rebuilt_i2000_teacher_consistency.json"
PACKAGE = REPRO / "autodl_47_segformer_rebuilt"
KAN_SHA = "c84d5c884a17fff40257b137b08ed08bc1729243a9884439b315b8132ccdb9af"

class StopQueue(RuntimeError): pass
def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""): value.update(chunk)
    return value.hexdigest()
def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); temp.replace(path)
def log(message):
    line=f"[{now()}] {message}"; print(line,flush=True)
    with MASTER_LOG.open("a",encoding="utf-8") as handle: handle.write(line+"\n")
def gpu_memory():
    try:
        output=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL,timeout=5)
        return max(int(x.strip()) for x in output.splitlines() if x.strip())
    except Exception: return None

def render_report():
    pl=STATE["pl"]; dist=STATE["distill"]; pkg=STATE["stage47_package"]
    lines=["# 历史PL → C6教师蒸馏 → 47 AutoDL包：夜间最终报告","",
      f"- 更新时间：`{STATE.get('updated_at')}`","- 流程状态：`%s`；停止原因：`%s`"%(STATE.get("status"),STATE.get("reason")),"",
      "## 固化的新C6教师锚","",f"- 状态：`{STATE['anchors'].get('status')}`；目录：`{ANCHORS}`",f"- 文件：`{STATE['anchors'].get('files')}`；manifest：`{ANCHOR_MANIFEST}`",
      f"- C6 checkpoint SHA-256：`{C6_CKPT_SHA}`；与历史锚像素一致率：`{STATE['anchors'].get('historical_pixel_agreement')}`。","- 历史D盘锚没有复制、没有进入训练，也没有用于后处理。","",
      "## 历史PL","",f"- 完成：`{pl.get('complete')}`；固定候选：`{PL_SNAPS[1]}`",f"- i2000 n500 plain EMA：`{pl.get('i2000_n500_miou')}`；各类别IoU：`{pl.get('i2000_per_class_iou')}`",
      f"- 权重SHA-256：`{pl.get('weights')}`",f"- 训练耗时秒：`{pl.get('duration_seconds')}`；峰值显存MiB：`{pl.get('peak_gpu_memory_mib')}`；中断/续训：`False/False`。","",
      "## C6教师硬标签蒸馏","",f"- 完成：`{dist.get('complete')}`；固定候选：`{DIST_SNAPS[1]}`",f"- i2000 n500 plain EMA：`{dist.get('i2000_n500_miou')}`；各类别IoU：`{dist.get('i2000_per_class_iou')}`",
      f"- 与新C6锚像素一致率：`{dist.get('teacher_pixel_agreement')}`；test1 ms2预测占比：`{dist.get('student_prediction_fraction')}`",f"- 权重SHA-256：`{dist.get('weights')}`",f"- 训练耗时秒：`{dist.get('duration_seconds')}`；峰值显存MiB：`{dist.get('peak_gpu_memory_mib')}`；中断/续训：`False/False`。","",
      "## 47 AutoDL复现包","",f"- 状态：`{pkg.get('status')}`；路径：`{pkg.get('path')}`",f"- 大小字节：`{pkg.get('bytes')}`；文件数：`{pkg.get('files')}`；manifest：`{pkg.get('manifest')}`",f"- 本地独立冒烟：`{pkg.get('smoke_status')}`；batch1、iter1、非候选；峰值显存MiB：`{pkg.get('smoke_peak_mib')}`。",
      "- 正式47冻结为 crop1024、真实batch8、seed17、4000 iter、restart LR .25、tau.85、teacher hflip、ignore-anchor-conflict、source_supervised .20、EMA、最终ms2。",
      "- 推荐32GB GPU；预计峰值19–20GiB；正式训练约1.5–3小时。AutoDL公开页当前RTX 5090 32GB标价约2.78元/小时，训练计算约4.17–8.34元，含准备/下载预留2–4小时约5.56–11.12元；启动前必须复核实时价格。","",
      "## 当前边界","","- 当前分类：本地候选。","- test2包：未生成。","- 已输出：否。","- 本轮真实外部指标：无。","- 正式47、49、test2、AutoDL和自动关机均未启动；本地电脑未关机。"]
    temp=REPORT.with_suffix(REPORT.suffix+".tmp"); temp.write_text("\n".join(lines)+"\n",encoding="utf-8"); temp.replace(REPORT)
def update(): STATE["updated_at"]=now(); write_json(STATUS,STATE); render_report()

def preflight():
    LOGS.mkdir(parents=True,exist_ok=True)
    if STATUS.exists() or REPORT.exists() or MASTER_LOG.exists(): raise StopQueue("overnight queue already exists")
    expected_absent=[ANCHORS,ANCHOR_MANIFEST,PL_MASKS,PACKAGE]+[STRICT/x for x in [PL_MAIN,PL_EVAL,*PL_SNAPS,DIST_MAIN,DIST_EVAL,CONSISTENCY,*DIST_SNAPS]]
    existing=[str(x) for x in expected_absent if x.exists()]
    if existing: raise StopQueue("refusing to overwrite: "+"; ".join(existing))
    if sha256(BASE)!=BASE_SHA or sha256(C6_CKPT)!=C6_CKPT_SHA or sha256(SOURCE/"kan_swin.py")!=KAN_SHA: raise StopQueue("base/C6/KAN source hash mismatch")
    repair=json.loads((TEACHER/"inference_path_repair_status.json").read_text(encoding="utf-8"))
    if repair.get("status")!="complete" or repair["identity"].get("closer")!="C6": raise StopQueue("C6 identity repair not complete/selected")
    if any(repair["boundaries"].get(k) for k in ("kan_student_trained","pl_started","distill_started","autodl_started","test2_inference_started")): raise StopQueue("prior boundary state is not clean")

def solidify_anchors():
    validation=json.loads(C6_VALIDATION.read_text(encoding="utf-8")); identity=json.loads(IDENTITY.read_text(encoding="utf-8"))
    comparison=identity["comparisons"]["historical_vs_c6"]
    if validation.get("status")!="valid" or validation.get("files")!=500 or comparison["pixel_agreement"]<0.90: raise StopQueue("rebuilt C6 anchor validation/identity gate failed")
    source=sorted(C6_PRED.glob("*.png")); test=sorted((REPRO/"test_1"/"images").glob("*.png"))
    if len(source)!=500 or {x.name for x in source}!={x.name for x in test}: raise StopQueue("C6/test1 file set mismatch")
    hashes={}; class_pixels=np.zeros(9,dtype=np.int64)
    for path in source:
        with Image.open(path) as opened: array=np.asarray(opened)
        if array.ndim!=2 or array.shape!=(1024,1024) or array.min()<1 or array.max()>8: raise StopQueue(f"invalid C6 anchor {path.name}")
        digest=sha256(path)
        if validation["sha256"].get(path.name)!=digest: raise StopQueue(f"C6 validation hash mismatch {path.name}")
        hashes[path.name]=digest; class_pixels+=np.bincount(array.ravel(),minlength=9)[:9]
    ANCHORS.mkdir(parents=True,exist_ok=False)
    for path in source:
        destination=ANCHORS/path.name; shutil.copy2(path,destination)
        if sha256(destination)!=hashes[path.name]: raise StopQueue(f"C6 anchor copy mismatch {path.name}")
    manifest={"status":"local_rebuilt_c6_teacher_anchors","created_at":now(),"files":500,"dimensions":[1024,1024],"channels":1,"allowed_labels":list(range(1,9)),
      "source_prediction_dir":str(C6_PRED),"anchor_dir":str(ANCHORS),"teacher_checkpoint":str(C6_CKPT),"teacher_checkpoint_sha256":C6_CKPT_SHA,
      "teacher_identity":"C6","historical_pixel_agreement":comparison["pixel_agreement"],"historical_identical_files":comparison["identical_png_pixels_files"],
      "historical_anchor_role":"read-only prior audit only; no historical file copied or used for training","inference":{"tile":1024,"overlap":0,"scales":[.75,1.,1.25],"horizontal_flip":True,"aggregation":"softmax_probability_average"},
      "class_pixels":{str(i):int(class_pixels[i]) for i in range(1,9)},"file_sha256":hashes}
    write_json(ANCHOR_MANIFEST,manifest); STATE["anchors"]={"status":"complete","files":500,"manifest":str(ANCHOR_MANIFEST),"historical_pixel_agreement":comparison["pixel_agreement"]}; update()

def audit_base_cache():
    import strict_env as P
    payload=P.load_checkpoint(BASE,map_location="cpu"); meta=payload["strict_repro"]
    if payload.get("iter")!=16000 or meta.get("stage")!="kan_from_v4_1_7439_16k" or payload.get("ema") is None: raise StopQueue("KAN16k lineage mismatch")
    del payload; gc.collect()
    manifest=json.loads((CACHE/"_CACHE_MANIFEST.json").read_text(encoding="utf-8")); confidence=json.loads((CACHE/"_CONFIDENCE_REPORT.json").read_text(encoding="utf-8"))
    if manifest!={"checkpoint":BASE.name,"checkpoint_sha256":BASE_SHA,"route":"ms2","source":"official_test1","expected_images":500}: raise StopQueue("PL cache manifest mismatch")
    files=sorted(CACHE.glob("*.npz")); names={x.stem for x in files}; tests={x.stem for x in (REPRO/"test_1"/"images").glob("*.png")}
    if len(files)!=500 or names!=tests: raise StopQueue("PL cache/test1 mismatch")
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={"pred","conf"} or z["pred"].shape!=(1024,1024) or z["conf"].shape!=(1024,1024) or not np.isfinite(z["conf"]).all(): raise StopQueue(f"invalid PL cache {path.name}")
    STATE["cache"]={"status":"passed","coverage_tau_085":confidence["coverage"]["0.85"],"warning":"60.8063% vs historical ~88.2%; warning only, tau fixed at .85"}; update()

def run_job(key,args,section):
    stdout=LOGS/f"overnight47_{key}.stdout.log"; stderr=LOGS/f"overnight47_{key}.stderr.log"; meta=STRICT/f"overnight47_{key}.json"
    if any(x.exists() for x in (stdout,stderr,meta)): raise StopQueue(f"job artifacts already exist: {key}")
    command=[str(PYTHON),"-u",*args]; baseline=gpu_memory(); peak=baseline; started=time.time()
    item={"status":"running","started_at":now(),"command":command,"stdout":str(stdout),"stderr":str(stderr),"resumed":False}; write_json(meta,item)
    STATE["current_stage"]=key; STATE[section]["status"]=key; update(); log("START "+key)
    with stdout.open("w",encoding="utf-8") as out, stderr.open("w",encoding="utf-8") as err:
        process=subprocess.Popen(command,cwd=str(SOURCE),stdout=out,stderr=err,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)); item["pid"]=process.pid; write_json(meta,item)
        while process.poll() is None:
            used=gpu_memory(); peak=used if peak is None else (max(peak,used) if used is not None else peak); time.sleep(10)
    item.update({"status":"complete" if process.returncode==0 else "failed","return_code":process.returncode,"ended_at":now(),"duration_seconds":time.time()-started,"peak_gpu_memory_mib":peak}); write_json(meta,item); log(f"END {key} exit={process.returncode}")
    if process.returncode!=0: raise StopQueue(f"{key} failed; see {stderr}")
    return item

def checkpoint_info(name,stage,iteration):
    import strict_env as P
    path=STRICT/name; payload=P.load_checkpoint(path,map_location="cpu")
    if payload.get("iter")!=iteration or payload.get("strict_repro",{}).get("stage")!=stage or payload.get("ema") is None: raise StopQueue(f"checkpoint lineage mismatch {name}")
    del payload; gc.collect(); return {"filename":name,"sha256":sha256(path),"bytes":path.stat().st_size,"iter":iteration,"stage":stage}
def eval_gate(result):
    score=result.get("val_mIoU"); values=result.get("val_IoU",{})
    if score is None or not math.isfinite(float(score)) or float(score)<.7300: return False,f"mIoU {score} < .7300/nonfinite"
    for name,value in values.items():
        if value is None or not math.isfinite(float(value)) or float(value)<=0: return False,f"{name} zero/nonfinite"
        if float(value)<.05: return False,f"{name} collapsed below .05"
    return True,"passed"

def run_pl():
    item=run_job("pl_train",["kan_swin.py","--train","--pl","--crop","640","--batch-size","2","--num-workers","3","--target-iters","4000","--max-seconds","86400","--seed","42","--stage","kan_historical_pl_rebuilt","--ckpt-name",PL_MAIN,"--warm-base",BASE.name,"--warm-parent-stage","kan_from_v4_1_7439_16k","--warm-parent-iter","16000","--pl-cache",CACHE.name,"--pl-tag","historical_rebuilt","--pl-tau","0.85","--restart-lr","0.25"],"pl")
    # The Windows child inherits the active Chinese code page.  The marker is
    # ASCII, so inspect raw bytes and avoid decoding unrelated Chinese text.
    if b"[resume]" in (LOGS/"overnight47_pl_train.stdout.log").read_bytes(): raise StopQueue("PL unexpectedly resumed")
    weights=[checkpoint_info(PL_MAIN,"kan_historical_pl_rebuilt",4000)]+[checkpoint_info(name,"kan_historical_pl_rebuilt",it) for name,it in zip(PL_SNAPS,(1000,2000,3000,4000))]
    STATE["pl"].update({"complete":True,"weights":weights,"duration_seconds":item["duration_seconds"],"peak_gpu_memory_mib":item["peak_gpu_memory_mib"]}); update()
    run_job("pl_i2000_eval_n500",["strict_eval.py","--model","kan","--ckpt",PL_SNAPS[1],"--n","500","--output",PL_EVAL,"--expected-stage","kan_historical_pl_rebuilt","--expected-iter","2000"],"pl")
    result=json.loads((STRICT/PL_EVAL).read_text(encoding="utf-8")); passed,reason=eval_gate(result); STATE["pl"].update({"status":"gate_passed" if passed else "gate_failed","i2000_n500_miou":result["val_mIoU"],"i2000_per_class_iou":result["val_IoU"],"gate_reason":reason}); update()
    if not passed: raise StopQueue("PL gate failed: "+reason)

def run_distill():
    item=run_job("distill_train",["kan_swin.py","--train","--anchor","--crop","640","--batch-size","2","--num-workers","3","--target-iters","4000","--max-seconds","86400","--seed","42","--stage","kan_segformer_distill_rebuilt","--ckpt-name",DIST_MAIN,"--warm-base",PL_SNAPS[1],"--warm-parent-stage","kan_historical_pl_rebuilt","--warm-parent-iter","2000","--anchor-dir",ANCHOR_NAME,"--restart-lr","0.25"],"distill")
    if b"[resume]" in (LOGS/"overnight47_distill_train.stdout.log").read_bytes(): raise StopQueue("distill unexpectedly resumed")
    weights=[checkpoint_info(DIST_MAIN,"kan_segformer_distill_rebuilt",4000)]+[checkpoint_info(name,"kan_segformer_distill_rebuilt",it) for name,it in zip(DIST_SNAPS,(1000,2000,3000,4000))]
    STATE["distill"].update({"complete":True,"weights":weights,"duration_seconds":item["duration_seconds"],"peak_gpu_memory_mib":item["peak_gpu_memory_mib"]}); update()
    run_job("distill_i2000_eval_n500",["strict_eval.py","--model","kan","--ckpt",DIST_SNAPS[1],"--n","500","--output",DIST_EVAL,"--expected-stage","kan_segformer_distill_rebuilt","--expected-iter","2000"],"distill")
    result=json.loads((STRICT/DIST_EVAL).read_text(encoding="utf-8")); passed,reason=eval_gate(result); STATE["distill"].update({"status":"eval_gate_passed" if passed else "eval_gate_failed","i2000_n500_miou":result["val_mIoU"],"i2000_per_class_iou":result["val_IoU"],"gate_reason":reason}); update()
    if not passed: raise StopQueue("distill gate failed: "+reason)
    run_job("distill_i2000_c6_consistency",["audit_kan_segformer_teacher_consistency.py","--ckpt",DIST_SNAPS[1],"--anchor-dir",ANCHOR_NAME,"--output",CONSISTENCY,"--expected-stage","kan_segformer_distill_rebuilt","--expected-iter","2000"],"distill")
    audit=json.loads((STRICT/CONSISTENCY).read_text(encoding="utf-8")); STATE["distill"].update({"status":"complete","teacher_pixel_agreement":audit["pixel_agreement"],"student_prediction_fraction":audit["student_prediction_fraction"],"consistency_report":CONSISTENCY}); update()

def prepare_package():
    item=run_job("prepare_stage47_package_and_smoke",["prepare_autodl47_package.py","--package",str(PACKAGE)],"stage47_package")
    manifest=json.loads((PACKAGE/"MANIFEST_SHA256.json").read_text(encoding="utf-8")); smoke=manifest["smoke"]
    STATE["stage47_package"]={"status":"complete","path":str(PACKAGE),"bytes":manifest["total_bytes_excluding_manifest"],"files":len(manifest["files"]),"manifest":str(PACKAGE/"MANIFEST_SHA256.json"),"smoke_status":smoke["status"],"smoke_peak_mib":smoke["peak_cuda_allocated_mib"],"duration_seconds":item["duration_seconds"]}; update()

STATE={"protocol":"overnight_segformer_pl_distill_47_20260926","started_at":now(),"updated_at":None,"status":"initializing","current_stage":None,"reason":None,"anchors":{"status":"not_started"},"cache":{"status":"not_started"},"pl":{"status":"not_started","complete":False},"distill":{"status":"not_started","complete":False},"stage47_package":{"status":"not_started"},"boundaries":{"local_candidate":True,"autodl_started":False,"formal_stage47_started":False,"stage49_started":False,"test2_inference":False,"test2_package":False,"submitted":False,"real_result_score":None}}
def main():
    preflight(); STATE.update({"status":"running","current_stage":"solidify_c6_anchors"}); update()
    try:
        solidify_anchors(); audit_base_cache(); run_pl(); run_distill(); prepare_package(); STATE.update({"status":"complete","current_stage":None,"reason":None,"finished_at":now()}); update(); log("COMPLETE; stopped before AutoDL/formal47")
    except StopQueue as error:
        STATE.update({"status":"stopped","current_stage":None,"reason":str(error),"finished_at":now()}); update(); log("STOP: "+str(error))
    except Exception as error:
        STATE.update({"status":"failed","current_stage":None,"reason":repr(error),"traceback":traceback.format_exc(),"finished_at":now()}); update(); log("FAILED: "+repr(error)); raise
if __name__=="__main__": main()
