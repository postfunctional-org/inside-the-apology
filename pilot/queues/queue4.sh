#!/bin/sh
# After queue3 (Llama from E3): Gemma 3 27B from the pain axis on. Log: out/queue4.log
cd "$(dirname "$0")/.."
while pgrep -f "sh queue2.sh" > /dev/null || pgrep -f "sh queue3.sh" > /dev/null; do sleep 60; done
sh chain.sh gemma-3-27b-it p1
echo "$(date -Is) queue4 done"
