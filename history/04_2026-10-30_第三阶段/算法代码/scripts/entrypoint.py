"""Commands shared by local runs and Docker."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    if len(sys.argv)<2 or sys.argv[1] not in {"verify","reproduce","infer","evaluate"}:
        raise SystemExit("Usage: entrypoint.py verify | reproduce | infer [options] | evaluate [options]")
    command=sys.argv[1]
    if command=="verify":
        argv=[sys.executable,str(ROOT/"code"/"prepare_inputs.py"),"--root",str(ROOT),"--verify-only"]
    elif command=="reproduce":
        argv=[sys.executable,str(ROOT/"scripts"/"reproduce_all_from_public_weights.py"),"--archive-root",str(ROOT)]+sys.argv[2:]
    else:
        argv=[sys.executable,str(ROOT/"code"/"kan_pipeline"/"final_pipeline.py"),"--archive-root",str(ROOT),command]+sys.argv[2:]
    raise SystemExit(subprocess.call(argv))
if __name__=="__main__": main()
