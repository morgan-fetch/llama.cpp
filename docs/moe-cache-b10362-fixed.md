# moe-cache-b10362-fixed

Branch: `moe-cache-b10362-fixed`

Working branch for fixing the MoE cache issue (b10362), based on
`moe-cache-b10362-rebased` (which carries the `feat: fate moe cache` commit).

Purpose: iterate on the MoE cache implementation and test it. To make local
testing possible, the flake gained a usable devShell (build tools such as
cmake and a C/C++ toolchain are available via `nix develop`).

The default devShell is CUDA-enabled (nvcc, cudart and cublas come from the
`cuda` package), because the FATE code only builds with `GGML_CUDA=ON`:

```console
$ nix develop
$ cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
$ cmake --build build
```

(`CMAKE_CUDA_ARCHITECTURES=native` picks the arch of the GPU on the machine,
e.g. `86` for an RTX 3060.)
