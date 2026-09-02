#!/usr/bin/env bash
# Builds dim_odom-<version>-py3-none-any.whl, the placeholder that makes a plain
# `dim-odom` dependency installable everywhere.
#
# Usage: ./build_fallback_wheel.sh [output dir, default target/wheels]
#
# cuVSLAM is proprietary, so there is no sdist to fall back on, and only three
# (platform, arch) slots have real wheels. Without this one, every other target
# has to be spelled out as an environment marker at each call site. The `any`
# platform tag sorts below every specific tag, so a real wheel always wins where
# one exists; this is only reached when nothing else matches.
set -euo pipefail
cd "$(dirname "$0")"

output=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${1:-target/wheels}")
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
mkdir -p "$build/dim_odom"

cat >"$build/pyproject.toml" <<EOF
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "dim_odom"
version = "$version"
description = "Visual-inertial odometry: cuVSLAM tracking plus error-state Kalman fusion"
license = { text = "Apache-2.0" }
requires-python = ">=3.10"

[tool.setuptools]
packages = ["dim_odom"]
EOF

cat >"$build/dim_odom/__init__.py" <<'EOF'
# Copyright 2026 Dimensional Inc.
# SPDX-License-Identifier: Apache-2.0

import platform
import sys

raise ImportError(
    "dim_odom ships no binary for %s/%s. The wheels cover linux x86_64, "
    "linux aarch64 and macOS arm64." % (sys.platform, platform.machine())
)
EOF

mkdir -p "$output"
python3 -m pip wheel --no-deps --wheel-dir "$output" "$build"
