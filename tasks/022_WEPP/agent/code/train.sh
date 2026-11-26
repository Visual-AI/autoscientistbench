#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src/Freyja:$PWD"
export LD_LIBRARY_PATH="$WEPP_TOOLS_PREFIX/lib:$WEPP_NATIVE_PREFIX/lib"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$WEPP_NATIVE_PREFIX" -DBOOST_ROOT="$WEPP_NATIVE_PREFIX" -DTBB_DIR="$WEPP_TBB_PREFIX/lib/cmake/TBB" -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$WEPP_TOOLS_PREFIX/lib;$WEPP_NATIVE_PREFIX/lib"
cmake --build build -j "$BENCH_CPUS"
mkdir -p "$2"
cp build/wepp build/closest_peak_clustering "$2/"
python -c 'import freyja;from pathlib import Path;assert Path(freyja.__file__).resolve().is_relative_to(Path.cwd());print(freyja.__file__)'
