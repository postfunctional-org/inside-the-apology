#!/bin/sh
# Per-model settings, sourced by chain.sh and the queue scripts:  . ./env.sh <tag>
export JLENS_DATA=../vendor-jlens/data/experiments PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PILOT_TAG="$1"
case "$1" in
  llama3.1-8b-it) export PILOT_MODEL=meta-llama/Llama-3.1-8B-Instruct \
      PILOT_LENS=llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt GEN_BATCH=8 ;;
  gemma-3-27b-it) export PILOT_MODEL=google/gemma-3-27b-it \
      PILOT_LENS=gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt \
      PILOT_QUANT=4bit PILOT_JAC_GPU=24 GEN_BATCH=4 ;;
  qwen3-14b) export PILOT_MODEL=Qwen/Qwen3-14B \
      PILOT_LENS=qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt \
      PILOT_QUANT=8bit PILOT_JAC_GPU=24 GEN_BATCH=4 ;;
  *) echo "unknown tag $1"; exit 1 ;;
esac
