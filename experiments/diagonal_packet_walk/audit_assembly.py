"""Inspect the actual Apple-clang assembly; static evidence, not dynamic tracing."""
import argparse
import hashlib
import json
import re
from collections import Counter

parser=argparse.ArgumentParser()
parser.add_argument("assembly")
parser.add_argument("--out",required=True)
args=parser.parse_args()
raw=open(args.assembly,"rb").read()
functions={}
for body in raw.decode().split("; -- End function"):
    names=re.findall(r'^([_A-Za-z][^ \t:]*):.*?@',body,re.M)
    if not names:continue
    name=names[-1]
    if "Lb0E" not in name and name not in ("__Z7product1ZS_","__Z7quarter1Zb"):continue
    instructions=Counter()
    vector_stack=[]
    for line in body.splitlines():
        m=re.match(r'\s+([a-z][a-z0-9.]*)\s',line)
        if m:instructions[m[1]]+=1
        if re.search(r'\b(?:ldr|str|ldp|stp|ldur|stur|ldnp|stnp)\s+[qds][0-9]+',line) and (
            re.search(r'\[(?:sp|x29)[,\]]',line) or "Spill" in line or "Reload" in line):
            vector_stack.append(line.strip())
    functions[name]={"static_instruction_counts":dict(instructions),
                     "vector_stack_accesses":vector_stack}
assert "__Z7product1ZS_" in functions
p=functions["__Z7product1ZS_"]["static_instruction_counts"]
assert p.get("ext.16b")==1 and p.get("fmul.2d")==1 and p.get("fmla.2d")==1,p
assert functions["__Z7quarter1Zb"]["static_instruction_counts"].get("ext.16b")==1
hot={k:v for k,v in functions.items() if "Lb0E" in k}
assert len(hot)>=3
assert all(not v["vector_stack_accesses"] for v in hot.values()),hot
with open(args.out,"w") as f:
    json.dump({"assembly_sha256":hashlib.sha256(raw).hexdigest(),
               "interpretation":"Static opcode and vector spill check; not dynamic instruction or cache traffic measurement",
               "functions":functions},f,indent=2)
    f.write("\n")
print("NEON primitive lowering verified; no vector stack accesses in",len(hot),"uncounted kernel functions")
