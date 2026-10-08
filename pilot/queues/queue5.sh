#!/bin/sh
# After queue3 (Llama from E3): finish Qwen3 14B from E2b, then Gemma 3 27B from the pain axis. Log: out/queue5.log
cd "$(dirname "$0")/.."
while pgrep -f "sh queue3.sh" > /dev/null; do sleep 60; done
sh chain.sh qwen3-14b s2b
sh chain.sh gemma-3-27b-it p1
echo "$(date -Is) queue5 done"
