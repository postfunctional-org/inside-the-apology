#!/bin/sh
# Qwen3 14B from E2b, Llama 3.1 8B from E3, then Gemma 3 27B from the pain axis. Log: out/queue6.log
cd "$(dirname "$0")/.."
sh chain.sh qwen3-14b s2b
sh chain.sh llama3.1-8b-it s3
sh chain.sh gemma-3-27b-it p1
echo "$(date -Is) queue6 done"
