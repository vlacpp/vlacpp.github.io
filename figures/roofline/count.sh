#!/bin/bash
# Counts FLOPs and stored-operand bytes for one predict() per model, at the benchmark's settings.
#
#   count.sh <vla-bench built against ggml-count.patch> <dir with one <id>.gguf per model>
#
# Real checkpoints or stub.py stand-ins both work. Each model runs with one call and with two;
# tally.py takes the difference, so loading and one-off setup graphs drop out.
set -u
bench=$1 dir=$2
mkdir -p logs
while IFS='|' read -r id args envs; do
  [ -f "$dir/$id.gguf" ] || { echo "$id: no $dir/$id.gguf, skipped"; continue; }
  for n in 1 2; do
    env VLA_FLOPCOUNT=1 $envs "$bench" --ckpt "$dir/$id.gguf" $args --warmup 0 --reps $n > "logs/$id.$n.log" 2>&1
    echo "$id reps=$n exit=$?"
  done
done <<'EOF'
smolvla|--images 2 --size 512|
octo|--images 2 --size 256|VLA_OCTO_UNNORM_DATASET=libero_object
turbo|--images 2 --size 256|
pi0|--images 2 --size 224|
pi05|--images 2 --size 224|
n15|--images 1 --size 224|VLA_GR00T_EMBODIMENT=new_embodiment
n16|--images 1 --size 224|VLA_GR00T_EMBODIMENT=libero_panda
n17|--images 1 --size 256|VLA_GR00T_EMBODIMENT=libero_sim
jepa|--images 1 --size 256 --extra-token 151697 --extra-count 32|
evo1|--images 1 --size 448|
adapter|--images 1 --size 224|
EOF
