#!/usr/bin/env bash
# Reproduces every training run of the report: ~2.5 h on 2 CPU cores (2 jobs in parallel).
#   bash scripts/run_experiments.sh            # ITERS=300 by default
#   ITERS=600 WORKERS=4 bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ITERS=${ITERS:-300}
WORKERS=${WORKERS:-2}
mkdir -p runs checkpoints

# 1) behaviour-cloning warm start of the shared actor from the Greedy-Track heuristic (~2 min)
[ -f checkpoints/bc_actor.pt ] || python -m marl.pretrain --out checkpoints/bc_actor.pt

# 2) MARL training. All learned methods start from the same BC actor and are trained against the
#    reliability targets tightened by a 10 % safety margin (evaluation always uses the nominal targets).
BC="init_actor=checkpoints/bc_actor.pt critic_warmup=5"
M="--env cost_margin=0.1"
cat > runs/jobs.txt << JOBS
--algo mappo_l --seed 0 --iters $ITERS $M --algo-cfg $BC
--algo mappo_l --seed 1 --iters $ITERS $M --algo-cfg $BC
--algo mappo_l --seed 2 --iters $ITERS $M --algo-cfg $BC
--algo mappo --seed 0 --iters $ITERS --algo-cfg $BC
--algo mappo_pen --seed 0 --iters $ITERS --penalty 1.0 $M --algo-cfg $BC
--algo ippo_l --seed 0 --iters $ITERS $M --algo-cfg $BC
--algo mappo_l --seed 0 --iters $ITERS --tag nomargin --algo-cfg $BC
--algo mappo_l --seed 0 --iters $ITERS --tag scratch $M
JOBS
# skip runs that already finished (final checkpoint present)
grep -v -e '^$' runs/jobs.txt | while read -r line; do
  algo=$(echo "$line" | awk '{print $2}'); seed=$(echo "$line" | awk '{print $4}')
  tag=$(echo "$line" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
  name="${algo}${tag:+_$tag}_s${seed}"
  [ -f "checkpoints/$name/final.pt" ] && echo "skip $name (done)" >&2 || echo "$line"
done > runs/jobs_todo.txt
nl -ba runs/jobs_todo.txt | xargs -P "$WORKERS" -L 1 sh -c 'n=$0; python scripts/train.py "$@" > "runs/log_job$n.txt" 2>&1'

# 3) extended training of two MAPPO-L seeds (300 -> 600 iterations; multipliers and critic are resumed)
for s in 0 1; do
  [ -f "checkpoints/mappo_l_long_s$s/final.pt" ] || echo "--algo mappo_l --seed $s --iters $ITERS --tag long $M --algo-cfg resume=checkpoints/mappo_l_s$s/final.pt"
done > runs/jobs_long.txt
nl -ba runs/jobs_long.txt | xargs -P "$WORKERS" -L 1 sh -c 'n=$0; python scripts/train.py "$@" > "runs/log_long$n.txt" 2>&1'
echo "all training jobs finished"
