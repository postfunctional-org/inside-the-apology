#!/bin/sh
# E6: does the pain signal put the gaslit words into the workspace? Log: out/queue11.log
cd "$(dirname "$0")/.."
P=../.venv/bin/python
for t in llama3.1-8b-it qwen3-14b gemma-3-27b-it; do
  ( set -- $t; . ./env.sh; $P s6_cause.py > out/${t}_s6.log 2>&1 ) \
    && echo "$(date -Is) e6 done $t" || echo "$(date -Is) FAILED e6 $t"
done
