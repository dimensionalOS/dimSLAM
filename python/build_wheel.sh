#!/usr/bin/env bash
# Builds a dim_odom wheel with the cuVSLAM runtime bundled in dim_odom/_libs.
#
# Usage: CUVSLAM_SDK_DIR=<sdk> ./build_wheel.sh [extra maturin args...]
#
# Linux wheels cross-compile from any host via zig, e.g.:
#   CUVSLAM_SDK_DIR=<sdk> ./build_wheel.sh --target aarch64-unknown-linux-gnu \
#       --zig --compatibility manylinux_2_39 --skip-auditwheel
# (--skip-auditwheel because the bundled libcuvslam is not a manylinux-policy lib;
# manylinux_2_39 matches the ubuntu24.04 the SDK binaries are built on. The SDK
# tarballs ship the .so under bin/, so symlink lib -> bin first.)
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

# zig rejects rustc's aarch64-linux Cortex-A53 erratum linker flag; route zig
# through a shim that strips it. maturin picks up whichever `zig` is first on PATH.
if command -v zig > /dev/null; then
    real_zig="$(command -v zig)"
    shim_dir="$(mktemp -d)"
    trap 'rm -rf dim_odom/_libs "$shim_dir"' EXIT
    cat > "$shim_dir/zig" << EOF
#!/bin/sh
for a in "\$@"; do
    shift
    case "\$a" in
        -Wl,--fix-cortex-a53-843419 | --fix-cortex-a53-843419) ;;
        *) set -- "\$@" "\$a" ;;
    esac
done
exec "$real_zig" "\$@"
EOF
    chmod +x "$shim_dir/zig"
    export PATH="$shim_dir:$PATH"
fi

maturin build --release "$@"
