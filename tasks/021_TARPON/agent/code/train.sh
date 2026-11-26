#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$2"
printf ready > "$2/ready"
