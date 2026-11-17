#!/usr/bin/env bash
# Commit anything new in the data repo (checked out at $DMT_DATA_DIR) and push it,
# rebasing on top of commits made meanwhile by the other data workflow.
# Usage: commit-data.sh "<message>"
set -euo pipefail
cd "${DMT_DATA_DIR:?DMT_DATA_DIR must point at the data repo checkout}"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
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
