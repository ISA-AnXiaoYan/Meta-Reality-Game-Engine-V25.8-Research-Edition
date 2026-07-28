#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:?usage: deploy_branch_snapshot.sh ARCHIVE [BRANCH] [COMMIT]}"
BRANCH="${2:-unknown}"
COMMIT="${3:-unknown}"
PROJECT="${PROJECT:-$PWD}"

cd "$PROJECT"
mkdir -p sync_ipc/remote_backups

TS="$(date +%Y%m%d_%H%M%S)"
FILELIST="/tmp/remote_ops_deploy_files_${TS}.txt"
EXISTING="/tmp/remote_ops_deploy_existing_${TS}.txt"
BACKUP="sync_ipc/remote_backups/git_branch_deploy_before_${TS}.tar.gz"

tar -tf "$ARCHIVE" > "$FILELIST"
while IFS= read -r file; do
  if [ -e "$file" ]; then
    echo "$file"
  fi
done < "$FILELIST" > "$EXISTING"

if [ -s "$EXISTING" ]; then
  tar -czf "$BACKUP" -T "$EXISTING"
else
  tar -czf "$BACKUP" --files-from /dev/null
fi

tar -xf "$ARCHIVE"

MARKER="GIT_BRANCH_DEPLOYED_REMOTE_OPS.txt"
{
  echo "branch=$BRANCH"
  echo "commit=$COMMIT"
  echo "deployed_at=$(date -Is)"
  echo "backup=$BACKUP"
  echo "archive=$ARCHIVE"
} > "$MARKER"

echo "backup=$BACKUP"
echo "marker=$MARKER"
cat "$MARKER"
