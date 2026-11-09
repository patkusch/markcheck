#!/usr/bin/env bash
# Download a bigger model pair for scripts/text_compare_pairs.py, resuming if it stops.
#
#   bash scripts/text_fetch_pair.sh Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct
#
# The usual Hugging Face download started over from zero when it stalled on a slow line.
# This keeps each part of the download and picks up where it stopped.
# Files go to ~/.cache/markcheck/models/<model name>; nothing goes into the repo.
set -u
M=~/.cache/markcheck/models
for repo in "$@"; do
  d=$M/$(basename "$repo" | tr '[:upper:]' '[:lower:]')
  mkdir -p "$d"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -s "$d/$f" ] || curl -sSfL -o "$d/$f" "https://huggingface.co/$repo/resolve/main/$f" || rm -f "$d/$f"
  done
  until curl -sSfL -C - -o "$d/model.safetensors" "https://huggingface.co/$repo/resolve/main/model.safetensors"; do
    sleep 5
  done
  echo "ready: $d"
done
