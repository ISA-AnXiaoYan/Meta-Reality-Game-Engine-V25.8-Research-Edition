#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-$SCRIPT_DIR/build}"

: "${CC:=/usr/bin/cc}"
: "${CXX:=/usr/bin/c++}"
export CC CXX

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$BUILD_DIR" -j"$(nproc)"
echo "$BUILD_DIR/hal_event_pipe_bridge"
echo "$BUILD_DIR/hal_event_fifo_broker"
echo "$BUILD_DIR/event_file_fifo_broker"
