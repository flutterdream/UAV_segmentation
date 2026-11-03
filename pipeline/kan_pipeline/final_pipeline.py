"""Portable single-KAN inference and test2 BN alpha075 calibration.
Fresh calibration takes only the newly trained i600 parent and official images.
Archived moments and predictions are evidence, never reconstruction inputs.
"""
import argparse, hashlib, json, os, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

MODEL_SHA = "60c499b9d860f2dd71ef964f047d4dbeed47a3b724f75f8a0b59a78bcf46952d"
PASSES = [(1.0, False), (1.0, True), (0.75, False), (0.75, True),
          (1.25, False), (1.25, True), (1.5, False), (0.5, False)]
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle"]

# ---- 扩展阶段扩展：测试集可通过环境变量切换（默认保持 test_2 / 1300，向后兼容）----
TEST_SPLIT = os.environ.get("AIC_TEST_SPLIT", "test_2")
TEST_COUNT = int(os.environ.get("AIC_TEST_COUNT", "1300"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def setup(root):
    root = Path(root).resolve()
    os.environ.update(STRICT_REPRO_ROOT=str(root), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(root / "code" / "kan_pipeline"))
    import strict_env as P
    import seg_swin as S
    import kan_swin as K
    # 推理模式下只用到测试图与最终权重，训练集/公开权重/派生统计不在本机，
    # 因此允许跳过 verify_inputs（权重正确性仍由 load_model 的 SHA 校验兜住）。
    if os.environ.get("AIC_SKIP_VERIFY", "0") != "1":
        P.verify_inputs()
    else:
        print("[verify] 已跳过 verify_inputs（AIC_SKIP_VERIFY=1，推理模式）", flush=True)
    torch.set_num_threads(8)
    return P,S,K

def load_model(root, checkpoint, shipped=False):
    P,S,K = setup(root); checkpoint=Path(checkpoint).resolve()
    if shipped and sha256(checkpoint) != MODEL_SHA:
        raise RuntimeError("Shipped winning checkpoint SHA mismatch")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    P.validate_checkpoint_payload(payload,checkpoint)
    if payload.get("iter") != 600 or payload.get("ema") is None:
        raise RuntimeError("Expected i600 EMA")
    if shipped and payload["strict_repro"]["stage"] != "kan_s2_stage47_i600_c1_alpha075_20261003":
        raise RuntimeError("Winning checkpoint stage mismatch")
    model=K.DeepKANSegSwin(K.NUM_CLASSES)
    model.load_state_dict(payload["ema"],strict=True)
    return payload,model.to(K.DEVICE).eval().requires_grad_(False),P,S,K

@torch.no_grad()
def predict_probability_average(model,image,S,device):
    height,width=image.shape[:2];accumulator=None
    for scale,flip in PASSES:
        transformed=image[:,::-1] if flip else image
        if scale != 1.0:
            transformed=np.asarray(Image.fromarray(transformed).resize(
                (int(width*scale),int(height*scale)),Image.BILINEAR))
        tensor=S.norm(transformed).unsqueeze(0).to(device)
        with torch.autocast("cuda",torch.float16,enabled=device=="cuda"):
            logits=model(tensor)
            if isinstance(logits,(tuple,list)): logits=logits[0]
        logits=logits.float()
        if scale != 1.0:
            logits=F.interpolate(logits,size=(height,width),mode="bilinear",align_corners=False)
        probabilities=logits.softmax(dim=1).cpu()
        if flip: probabilities=torch.flip(probabilities,dims=(3,))
        accumulator=probabilities if accumulator is None else accumulator+probabilities
        del tensor,logits,probabilities
    averaged=accumulator/len(PASSES)
    if not torch.isfinite(averaged).all() or (averaged.sum(1)-1).abs().max()>1e-5:
        raise RuntimeError("Invalid probability average")
    return averaged.squeeze(0)

def moments_hook(bucket):
    def hook(_module,inputs):
        tensor=inputs[0].detach().float()
        variance,mean=torch.var_mean(tensor,dim=(0,2,3),unbiased=False)
        mean,variance=mean.double().cpu(),variance.double().cpu()
        count=int(tensor.shape[0]*tensor.shape[2]*tensor.shape[3])
        if bucket["count"]==0:
            bucket.update(count=count,mean=mean,m2=variance*count)
        else:
            previous=bucket["count"];delta=mean-bucket["mean"];total=previous+count
            bucket["mean"]+=delta*(count/total)
            bucket["m2"]+=variance*count+delta.square()*previous*count/total
            bucket["count"]=total
    return hook

def calibrate(root,parent,output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    payload,model,P,S,K=load_model(root,parent)
    if payload["strict_repro"]["stage"]!="kan_stage47_earlystop_bracket":
        raise RuntimeError("Requires freshly reproduced Stage47 i600 parent")
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    bns={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)}
    if len(bns)!=41: raise RuntimeError("Expected 41 BN layers")
    buckets={n:{"count":0} for n in bns}
    hooks=[m.register_forward_pre_hook(moments_hook(buckets[n])) for n,m in bns.items()]
    files=sorted((Path(root)/"official_data"/TEST_SPLIT/"images").glob("*.png"))
    if len(files)!=TEST_COUNT: raise RuntimeError(f"Expected {TEST_COUNT} images in {TEST_SPLIT}, got {len(files)}")
    try:
        with torch.inference_mode():
            for i,path in enumerate(files,1):
                image=np.asarray(Image.open(path).convert("RGB"))
                if image.shape!=(1024,1024,3): raise RuntimeError(path.name)
                tensor=S.norm(image).unsqueeze(0).to(K.DEVICE)
                with torch.autocast("cuda",torch.float16,enabled=K.DEVICE=="cuda"): result=model(tensor)
                if not torch.isfinite(result).all(): raise RuntimeError("Non-finite output")
                if i%25==0: print(f"calibration {i}/{TEST_COUNT}",flush=True)
    finally:
        for hook in hooks: hook.remove()
    if not all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in original.items()):
        raise RuntimeError("Statistics collection mutated model")
    moments={};alpha=.75
    for name,module in bns.items():
        bucket=buckets[name]
        if bucket["count"]<=1: raise RuntimeError("Insufficient statistics")
        tm=bucket["mean"].float();tv=(bucket["m2"]/(bucket["count"]-1)).float()
        sm,sv=original[name+".running_mean"],original[name+".running_var"]
        mean=(1-alpha)*sm+alpha*tm
        var=(1-alpha)*sv+alpha*tv+alpha*(1-alpha)*(tm-sm).square()
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var<=0).any():
            raise RuntimeError("Invalid blended moments")
        with torch.no_grad():
            module.running_mean.copy_(mean);module.running_var.copy_(var)
        moments[name]={"source_mean":sm.tolist(),"source_var":sv.tolist(),
                       "target_mean":tm.tolist(),"target_var":tv.tolist(),"sample_count":bucket["count"]}
    calibrated={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    changed=[k for k in original if not torch.equal(original[k],calibrated[k])]
    expected={n+s for n in bns for s in (".running_mean",".running_var")}
    if len(changed)!=82 or set(changed)!=expected: raise RuntimeError("BN-only gate failed")
    output.parent.mkdir(parents=True,exist_ok=True)
    moments_path=output.with_suffix(".bn_moments.json");write_json(moments_path,moments)
    torch.save({"model":payload["model"],"ema":calibrated,"iter":600,
        "strict_repro":P.strict_metadata("kan_stage47_i600_test2_alpha075_retrained",17,[Path(parent).name]),
        "parent_sha256":sha256(parent),"bn_moments_sha256":sha256(moments_path),
        "changed_buffers":changed,"statistics_image_count":TEST_COUNT,
        "stage47_config":payload.get("stage47_config"),
        "stage49_config":{"alpha_mean":alpha,"alpha_var":alpha,"statistics_routes":1,
                          "gradient_updates":False,"variance_between_group_term":True}},output)
    split=json.loads((Path(root)/"code"/"kan_pipeline"/"split.json").read_text(encoding="utf-8"))
    miou,ious=S.evaluate(model,split["val_ids"],tta=False,bias=None)
    write_json(output.with_suffix(".eval.json"),{"n":500,"route":"plain_ema","miou":float(miou),
        "class_iou":dict(zip(CLASS_NAMES,map(float,ious))),"real_score":None})
    if not np.isfinite(miou) or miou<.74 or any(not np.isfinite(v) or v<=0 for v in ious):
        raise RuntimeError("Fixed local collapse gate failed")

def infer(root,checkpoint,output_dir,retrained=False,limit=None):
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError("Output must be empty")
    output_dir.mkdir(parents=True,exist_ok=True)
    _,model,P,S,K=load_model(root,checkpoint,shipped=not retrained)
    files=sorted((Path(root)/"official_data"/TEST_SPLIT/"images").glob("*.png"))
    if len(files)!=TEST_COUNT: raise RuntimeError(f"Expected {TEST_COUNT} images in {TEST_SPLIT}, got {len(files)}")
    selected=files[:limit] if limit else files
    pred_dir=output_dir/"predictions";pred_dir.mkdir();manifest=[]
    for i,path in enumerate(selected,1):
        image=np.asarray(Image.open(path).convert("RGB"))
        if image.shape!=(1024,1024,3): raise RuntimeError(path.name)
        pred=predict_probability_average(model,image,S,K.DEVICE).argmax(0).numpy().astype(np.uint8)
        if pred.shape!=(1024,1024) or pred.min()<1 or pred.max()>8: raise RuntimeError(path.name)
        dest=pred_dir/path.name;Image.fromarray(pred).save(dest)
        manifest.append({"name":path.name,"sha256":sha256(dest)})
        if i%25==0 or i==len(selected): print(f"inference {i}/{len(selected)}",flush=True)
    archive=output_dir/("smoke_predictions.zip" if limit else "predictions.zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in selected: z.write(pred_dir/path.name,arcname=path.name)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!={p.name for p in selected} or z.testzip() is not None:
            raise RuntimeError("ZIP integrity failure")
    write_json(output_dir/"INFERENCE_REPORT.json",{"model_sha256":sha256(checkpoint),
        "model_state":"ema","single_model":True,"image_count":len(selected),"passes":PASSES,
        "aggregation":"probability_average","files":manifest,"zip_sha256":sha256(archive),
        "smoke_only":bool(limit),"submitted":False,"real_score":None})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive-root",type=Path,default=Path(__file__).resolve().parents[2])
    sub=parser.add_subparsers(dest="command",required=True)
    c=sub.add_parser("calibrate");c.add_argument("--parent",type=Path,required=True);c.add_argument("--output",type=Path,required=True)
    i=sub.add_parser("infer");i.add_argument("--checkpoint",type=Path,required=True);i.add_argument("--output-dir",type=Path,required=True)
    i.add_argument("--retrained",action="store_true");i.add_argument("--limit",type=int)
    e=sub.add_parser("evaluate");e.add_argument("--checkpoint",type=Path,required=True);e.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="calibrate": calibrate(args.archive_root,args.parent,args.output)
    elif args.command=="infer":
        if args.limit is not None and not (1<=args.limit<=TEST_COUNT): parser.error(f"--limit must be 1..{TEST_COUNT}; smoke-only")
        infer(args.archive_root,args.checkpoint,args.output_dir,args.retrained,args.limit)
    else:
        _,model,_,S,_=load_model(args.archive_root,args.checkpoint,shipped=True)
        split=json.loads((args.archive_root/"code"/"kan_pipeline"/"split.json").read_text(encoding="utf-8"))
        miou,ious=S.evaluate(model,split["val_ids"],tta=False,bias=None)
        write_json(args.output,{"n":500,"miou":float(miou),"class_iou":dict(zip(CLASS_NAMES,map(float,ious))),"real_score":None})

if __name__=="__main__": main()
