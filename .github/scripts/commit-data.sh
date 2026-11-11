#!/usr/bin/env bash
# Commit anything new under data/ and push, rebasing on top of commits made
# meanwhile by the other data workflows. Usage: commit-data.sh "<message>"
set -euo pipefail
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add data
if git diff --cached --quiet; then
  echo "No new data."
  exit 0
fi
git commit -q -m "$1"
for attempt in 1 2 3 4 5; do
  if git pull --rebase -q && git push -q; then
    exit 0
  fi
  echo "Push failed (attempt $attempt), retrying"
  sleep $((attempt * 10))
done
exit 1
