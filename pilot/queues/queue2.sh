#!/bin/sh
# Gemma 3 27B, then Qwen3 14B, after the Llama chain that is already running. Log: out/queue2.log
cd "$(dirname "$0")/.."
while [ ! -f out/llama3.1-8b-it_chain.done ] && pgrep -f "chain.sh llama3.1-8b-it" > /dev/null; do sleep 30; done
sh chain.sh gemma-3-27b-it
while pgrep -f "hf download Qwen/Qwen3-14B" > /dev/null || pgrep -f "include qwen3-14b" > /dev/null; do sleep 60; done
sh chain.sh qwen3-14b
echo "$(date -Is) queue done"
