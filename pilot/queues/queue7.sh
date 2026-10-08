#!/bin/sh
# Clamp conditions again for Llama and Qwen3 14B with the sink-safe clamp (out/<tag>_e3clamp.json), then
# Gemma 3 27B from E3 with the sink-safe clamp and the capped dose. Log: out/queue7.log
cd "$(dirname "$0")/.."
P=../.venv/bin/python
for t in llama3.1-8b-it qwen3-14b; do
  ( set -- "$t"; . ./env.sh; E3_ONLY=abl_pain,abl_rest,abl_randdir E3_NAME=e3clamp $P s3_split.py > out/${t}_s3clamp.log 2>&1 ) \
    && echo "$(date -Is) clamp rerun done $t" || echo "$(date -Is) FAILED clamp rerun $t"
done
sh chain.sh gemma-3-27b-it s3
echo "$(date -Is) queue7 done"
