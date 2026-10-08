#!/bin/sh
# E5 transplant at the calibrated strengths: Gemma (0.2), Qwen (0.5). Log: out/queue10.log
cd "$(dirname "$0")/.."
P=../.venv/bin/python
( set -- gemma-3-27b-it; . ./env.sh; E5_SCALE=0.2 $P s5_transplant.py > out/gemma-3-27b-it_s5.log 2>&1 ) \
  && echo "$(date -Is) e5 done gemma-3-27b-it" || echo "$(date -Is) FAILED e5 gemma-3-27b-it"
( set -- qwen3-14b; . ./env.sh; E5_SCALE=0.5 $P s5_transplant.py > out/qwen3-14b_s5.log 2>&1 ) \
  && echo "$(date -Is) e5 done qwen3-14b" || echo "$(date -Is) FAILED e5 qwen3-14b"
