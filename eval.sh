#!/bin/bash
# eval.sh — Full benchmark: 3 orders × 3 seeds = 9 runs
#
# Best q+v settings: Fisher-weighted merge (beta=0.5), no EWC penalty, no LoRA+
#
# Usage:
#   bash eval.sh                    # run all 9 sequentially
#   bash eval.sh O4 42              # run a single order+seed

set -euo pipefail

BASE_CMD="python train.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --method slao \
    --optimizer adamw \
    --lr 1e-4 \
    --batch_size 8 \
    --grad_accum 1 \
    --epochs 1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules q_proj v_proj \
    --samples_per_class_train 1000 \
    --samples_per_class_val 500 \
    --max_length 512 \
    --torch_dtype bfloat16 \
    --eval_batch_size 32 \
    --max_new_tokens 5 \
    --fisher_merge_beta 0.5"

ORDERS=("O4" "O5" "O6")
SEEDS=("42" "51" "60")

run_one() {
    local order=$1
    local seed=$2
    local outdir="outputs/bench_fmerge05_${order}_s${seed}"

    if [ -f "${outdir}/results.json" ]; then
        echo "=== SKIP ${order} seed=${seed} (already done) ==="
        return
    fi

    echo "=== START ${order} seed=${seed} ==="
    $BASE_CMD \
        --task_order "$order" \
        --seed "$seed" \
        --output_dir "$outdir"
    echo "=== DONE ${order} seed=${seed} ==="
}

# Allow running a single order+seed: bash eval.sh O4 42
if [ $# -eq 2 ]; then
    run_one "$1" "$2"
    exit 0
fi

# Run all 9
for order in "${ORDERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one "$order" "$seed"
    done
done

# Collect results
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo ""
printf "%-25s %8s %8s\n" "Run" "AA" "BWT"
printf "%-25s %8s %8s\n" "---" "---" "---"
for order in "${ORDERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        d="outputs/bench_fmerge05_${order}_s${seed}"
        if [ -f "$d/results.json" ]; then
            printf "%-25s " "${order}_s${seed}"
            python3 -c "
import json
r = json.load(open('$d/results.json'))
print(f'{r[\"average_accuracy\"]:8.4f} {r[\"backward_transfer\"]:8.4f}')
"
        fi
    done
done

echo ""
echo "Per-order averages:"
for order in "${ORDERS[@]}"; do
    python3 -c "
import json, numpy as np
aas, bwts = [], []
for seed in ['42', '51', '60']:
    path = f'outputs/bench_fmerge05_${order}_s{seed}/results.json'
    try:
        r = json.load(open(path))
        aas.append(r['average_accuracy'])
        bwts.append(r['backward_transfer'])
    except FileNotFoundError:
        pass
if aas:
    print(f'  ${order}: AA={np.mean(aas):.4f}±{np.std(aas):.4f}  BWT={np.mean(bwts):.4f}±{np.std(bwts):.4f}  (n={len(aas)})')
else:
    print(f'  ${order}: no results yet')
"
done

echo ""
echo "Overall average:"
python3 -c "
import json, numpy as np
aas, bwts = [], []
for order in ['O4', 'O5', 'O6']:
    for seed in ['42', '51', '60']:
        path = f'outputs/bench_fmerge05_{order}_s{seed}/results.json'
        try:
            r = json.load(open(path))
            aas.append(r['average_accuracy'])
            bwts.append(r['backward_transfer'])
        except FileNotFoundError:
            pass
if aas:
    print(f'  AA={np.mean(aas):.4f}±{np.std(aas):.4f}  BWT={np.mean(bwts):.4f}±{np.std(bwts):.4f}  (n={len(aas)})')
else:
    print('  no results yet')
"
