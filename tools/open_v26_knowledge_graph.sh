#!/usr/bin/env sh
# SPDX-License-Identifier: AGPL-3.0-only
set -eu

viewer_root="${TMPDIR:-/tmp}/mrge-v26-knowledge-graph-viewer"
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
graph_root="$repo_root/historical/knowledge-graph/v26"

test -f "$graph_root/knowledge-graph.json"
test -f "$graph_root/meta.json"
mkdir -p "$viewer_root/.ua"
cp "$graph_root/knowledge-graph.json" "$viewer_root/.ua/knowledge-graph.json"
cp "$graph_root/meta.json" "$viewer_root/.ua/meta.json"
printf '%s\n' '{"outputLanguage":"zh"}' > "$viewer_root/.ua/config.json"

printf '%s\n' 'Starting the pinned public V26 graph viewer. It only serves local files.'
npx --yes "https://github.com/Egonex-AI/Understand-Anything/releases/download/v2.9.4/understand-anything-viewer.tgz" "$viewer_root"
