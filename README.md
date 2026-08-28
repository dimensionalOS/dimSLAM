# dimSLAM

VSLAM stack for dimos.

- `cuvslam/` — our cuVSLAM lineage (NVIDIA cuVSLAM v17 + odometry state
  serialization). Licensed under the NVIDIA Community License (`cuvslam/LICENSE`).
- `src/`, `CMakeLists.txt` — the dimos native odometry module (`cuvslam_odometry`)
  built on top of it.
- `flake.nix` — builds both: `nix build .#<variant>` (variants: `x86_64-cuda12`,
  `x86_64-cuda13`, `orin`, `thor`, `metal`; fork-source builds for x86_64-cuda12
  and orin, NVIDIA release tarballs for the rest). The dimos header-only native
  C++ SDK comes in as the pinned `dimos-src` flake input.

dimos consumes this repo by git tag:
`nix build github:dimensionalOS/dimSLAM/<tag>#<variant>`
