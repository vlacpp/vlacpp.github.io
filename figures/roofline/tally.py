#!/usr/bin/env python3
"""Per-predict() FLOPs and bytes from count.sh's logs: the two-call run minus the one-call run."""
import glob, re

for path in sorted(glob.glob("logs/*.1.log")):
    model = path.split("/")[-1].split(".")[0]
    runs = []
    for n in (1, 2):
        try:
            text = open(f"logs/{model}.{n}.log").read()
        except FileNotFoundError:
            break
        m = re.search(r"FLOPCOUNT graphs=\d+ mul_mat=\d+ mm_flop=(\S+) other_flop=(\S+) bytes=(\S+)", text)
        if not m:
            break
        runs.append((float(m[1]) + float(m[2]), float(m[3])))
    if len(runs) < 2:
        print(f"{model:8s} incomplete")
        continue
    flop, nbytes = runs[1][0] - runs[0][0], runs[1][1] - runs[0][1]
    print(f"{model:8s} {flop / 1e9:8.1f} GFLOP {nbytes / 1e9:7.3f} GB {flop / nbytes:7.1f} FLOP/byte")
