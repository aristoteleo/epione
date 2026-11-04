#!/usr/bin/env python
# Reference-side helper for MANIFEST.md (runs in the RGT environment).
"""Write each motif of a JASPAR file as an RGT .pwm file named <id>.<name>.pwm."""
import re, sys, os
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
name = None; rows = []
def flush():
    if name:
        with open(os.path.join(out, name + ".pwm"), "w") as f:
            for r in rows:
                f.write(" ".join(r) + "\n")
for line in open(src):
    line = line.strip()
    if not line:
        continue
    if line.startswith(">"):
        flush()
        mid, mname = line[1:].split(None, 1)
        name = f"{mid}.{mname.strip()}"; rows = []
    else:
        vals = re.findall(r"[-\d.eE]+", line.split("[", 1)[1])
        rows.append(vals)
flush()
