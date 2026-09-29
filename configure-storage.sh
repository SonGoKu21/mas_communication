#!/usr/bin/env bash
set -euo pipefail
code="${MAS_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
data="${MAS_DATA_ROOT:?Set MAS_DATA_ROOT to an absolute storage directory}"
[[ "$data" = /* ]] || { printf "MAS_DATA_ROOT must be absolute\n" >&2; exit 1; }
for name in models datasets results logs artifacts; do
    mkdir -p "$data/$name"
    target="$code/$name"
    if [[ -e "$target" || -L "$target" ]]; then
        if [[ ! -L "$target" || "$(readlink "$target")" != "$data/$name" ]]; then
            printf 'Refusing to replace existing path: %s\n' "$target" >&2
            exit 1
        fi
    else
        ln -s "$data/$name" "$target"
    fi
done
mkdir -p "$data/cache/modelscope" "$data/cache/xdg" "$data/tmp"
printf 'Storage paths configured under %s\n' "$data"
