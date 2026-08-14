# moe-cache-b10362-fixed

Branch: `moe-cache-b10362-fixed`

Working branch for fixing the MoE cache issue (b10362), based on
`moe-cache-b10362-rebased` (which carries the `feat: fate moe cache` commit).

Purpose: iterate on the MoE cache implementation and test it. To make local
testing possible, the flake gained a usable devShell (build tools such as
cmake and a C/C++ toolchain are available via `nix develop`).
