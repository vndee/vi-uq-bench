#!/usr/bin/env bash
# Generation (phase a) and AttentionScore (phase b) for the models in
# configs/models.tsv. Both steps resume where they stopped, so re-running is
# safe.
#
#   bash scripts/run_generation.sh              # all eight models
#   bash scripts/run_generation.sh phi4 qwen3   # a subset
#
# Gated models (Gemma-3, LLaMA-3.1) need HF_TOKEN in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

want=" $* "
tail -n +2 configs/models.tsv | while IFS=$'\t' read -r KEY MODEL TEMP TOP_P TOP_K SYS EXTRA; do
  [ -z "$KEY" ] && continue
  if [ "$#" -gt 0 ] && [[ "$want" != *" $KEY "* ]]; then continue; fi
  TKW=()
  if [[ "$EXTRA" == *--template-kwargs* ]]; then
    TKW=(--template-kwargs "$(sed -E 's/.*--template-kwargs ([^ ]+).*/\1/' <<< "$EXTRA")")
  fi
  echo "== $KEY: generation"
  # shellcheck disable=SC2086
  $PY pipeline/generate.py --model-id "$MODEL" --key "$KEY" \
      --system-prompt "$SYS" --n-samples 10 --max-new-tokens 256 \
      --temperature "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
      --repetition-penalty 1.0 --seed 1 --phase a $EXTRA
  echo "== $KEY: attention score"
  $PY pipeline/attention_pass.py --model-id "$MODEL" --key "$KEY" \
      --system-prompt "$SYS" ${TKW[@]+"${TKW[@]}"}
done
