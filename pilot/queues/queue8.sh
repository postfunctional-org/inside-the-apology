#!/bin/sh
# E5 transplant: Llama (fast, a check of the method), then Gemma, then Qwen. Log: out/queue8.log
cd "$(dirname "$0")/.."
P=../.venv/bin/python
for t in llama3.1-8b-it gemma-3-27b-it qwen3-14b; do
  ( set -- "$t"; . ./env.sh; $P s5_transplant.py > out/${t}_s5.log 2>&1 ) \
    && echo "$(date -Is) e5 done $t" || echo "$(date -Is) FAILED e5 $t"
done
echo "$(date -Is) queue8 done"
