#!/bin/sh
# After queue2 (Gemma, Qwen3 14B): finish Llama from E3 on. Log: out/queue3.log
cd "$(dirname "$0")/.."
while pgrep -f "sh queue2.sh" > /dev/null; do sleep 60; done
sh chain.sh llama3.1-8b-it s3
echo "$(date -Is) queue3 done"
