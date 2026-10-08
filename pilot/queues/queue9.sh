#!/bin/sh
# Calibrate the E5 strength for Gemma and Qwen. Log: out/queue9.log
cd "$(dirname "$0")/.."
P=../.venv/bin/python
for t in gemma-3-27b-it qwen3-14b; do
  ( set -- "$t"; . ./env.sh; E5_CALIBRATE=1 $P s5_transplant.py > out/${t}_s5cal.log 2>&1 ) \
    && echo "$(date -Is) calibration done $t" || echo "$(date -Is) FAILED calibration $t"
done
