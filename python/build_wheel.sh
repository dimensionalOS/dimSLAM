#!/usr/bin/env bash
# Builds a dim_odom wheel with the cuVSLAM runtime bundled in dim_odom/_libs.
#
# Usage: CUVSLAM_SDK_DIR=<sdk> ./build_wheel.sh [extra maturin args...]
#
# Linux wheels must be built in a Linux environment with the GNU toolchain (zig
# cross-compiling links the shim against LLVM libc++, whose std:: symbols don't
# match the libstdc++-built libcuvslam), e.g. from a mac:
#   docker run --rm -v <repo>:/io -v <sdk>:/sdk ubuntu:24.04 bash -c '
#     apt-get update -qq && apt-get install -y -qq curl build-essential python3 python3-pip
#     curl -sSf https://sh.rustup.rs | sh -s -- -y -q && . $HOME/.cargo/env
#     pip install -q --break-system-packages maturin
#     CUVSLAM_SDK_DIR=/sdk /io/python/build_wheel.sh --compatibility manylinux_2_39 --skip-auditwheel'
# (--skip-auditwheel because the bundled libcuvslam is not a manylinux-policy lib;
# manylinux_2_39 matches the ubuntu24.04 the SDK binaries are built on. The SDK
# tarballs ship the .so under bin/, so symlink lib -> bin first. Use a
# --platform linux/amd64 or linux/arm64 image to pick the wheel arch.)
#
# _libs exists only for the duration of the build: maturin drops gitignored files
# from the wheel even when explicitly included, so the directory cannot simply be
# ignored, and leaving it around would dirty the tree. CuMetal finds cumetal-cache
# beside the libcumetal.dylib it was loaded from; the extension's rpath points at
# _libs (see build.rs).
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${CUVSLAM_SDK_DIR:-}" ]; then
    echo "CUVSLAM_SDK_DIR is not set" >&2
    exit 1
fi

trap 'rm -rf dim_odom/_libs' EXIT
rm -rf dim_odom/_libs
mkdir -p dim_odom/_libs
cp "$CUVSLAM_SDK_DIR"/lib/lib* dim_odom/_libs/
if [ -e "$CUVSLAM_SDK_DIR"/lib/libcumetal.dylib ]; then
    cp -R "$CUVSLAM_SDK_DIR"/share/cumetal-cache dim_odom/_libs/cumetal-cache
fi
chmod -R u+w dim_odom/_libs

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
maturin build --release "$@"
