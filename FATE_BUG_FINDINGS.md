# FATE broken-output findings (2026-08-14)

Reproducer and root-cause analysis for the scrambled logits (`//////` output)
seen when running llama.cpp's `--fate` MoE expert cache on a CUDA build.

## Reproducer

Environment: llama.cpp fork `moe-cache-b10362-fixed` (build `b10366-800a58f56`),
cmake CUDA build in `build-cuda/` (`GGML_CUDA=ON`, arch 86, `GGML_CUDA_GRAPHS=ON`),
RTX 3060 (CUDA0, 11894 MiB), model Qwen3.6 35B A3B Q4_K
(`/nix/store/73rpbsn9j6s58dk8q74309hvh0l1bphb-qwen-3-6-35b-a3b.gguf`).

Command (one-shot, exits cleanly):

```
./build-cuda/bin/llama-completion -m <model> -p "Hello" -n 16 -c 1024 \
    --fate --fate-cache 2048
```

Output — always this, regardless of sampler seed:

```
user
Hello
assistant
////////////////
```

Baseline without `--fate` (same binary): clean output, 37-38 t/s, exit 0.

## Observed FATE behavior (identical on every run)

```
FATE: n_layer=40 n_expert=256 n_expert_used=8
FATE: max expert strides: gate=589824B up=589824B down=860160B max=0.8MB
FATE: working set = 960 slots (788MB), target = 2048MB
FATE: GPU pool allocated: 2495 slots × 0.8MB = 2047MB
FATE: cudaHostRegister failed for all flags (ptr=... size=144MB)
FATE: pinned 0/120 expert tensors for async prefetch
FATE: pinned staging buffer: 0.8MB
FATE: prefetch engine started (cross-layer + temporal prediction)
FATE: system initialized (2495 cache slots + prefetch)
...
======== FATE CACHE STATS ========
  accesses   : 17577
  hits       : 17208 (D2D from pool)
  misses     : 369 (H2D fallback)
  hit rate   : 97.90%
  prefetched : 1299 (async H2D to pool)
  pool       : 2495 slots × 0.8MB
==================================
```

The pool mostly serves data (97.9% hit rate) and it serves *wrong* data.

## What is NOT the cause

- **bwrap / sandbox memlock limits.** The identical garbage reproduces on
  Sarah's machine without bubblewrap. `cudaHostRegister` failing for the model
  weights only disables the "truly async from pinned weights" fast path; the
  fallback copies are still *correct* (CUDA stages pageable→pinned internally
  at issue time). Pinning failure = performance issue, not corruption.
- **CUDA graphs.** The expert-copy hook runs in the scheduler's input-copy
  phase (every iteration), outside graph capture/replay. `--no-mmap`, `--no-cuda-graphs`
  variants still produce the same garbage.
- **Key/slot bookkeeping.** `make_key(layer,kind,expert)` is collision-free
  (256 experts fit exactly in 8 bits), slot sizes are consistent
  (`expert_size + min(expert_size,512)` everywhere), and D2D copies via
  `cudaMemcpyDefault` correctly recognize both pool and `input_cpy` as device
  memory.

## Root cause: single staging buffer reused for async copies without sync

`fate_system::on_expert_copy()` (src/llama-fate.cpp, layer-transition block)
prefetches predicted experts for the next layer into pool slots with this loop:

```c
if (prefetch.staging && copy_n <= prefetch.staging_size) {
    memcpy(prefetch.staging, src, copy_n);                              // CPU write
    fate_prefetch_h2d(prefetch.stream, dst_ptr, prefetch.staging, copy_n); // async H2D
}
```

`fate_prefetch_h2d()` is `cudaMemcpyAsync(..., cudaMemcpyDefault, stream)`.
For **pinned** host memory (the staging buffer is `cudaMallocHost`, which
succeeded), `cudaMemcpyAsync` is genuinely asynchronous: the copy engine DMAs
out of the staging buffer *after* the API returns. The loop immediately
overwrites that same single staging buffer with the next expert's data, so
every enqueued copy (except possibly the last in each batch) reads
already-overwritten memory. The pool slots get wrong expert weights; the hit
path (`ggml_backend_tensor_set_async(backend, dst, slot_ptr, ...)`) then serves
that garbage into the MoE compute — deterministic scrambled logits.

This explains every observation:

- happens with and without bwrap (machine-independent data race);
- happens with mmap and `--no-mmap` (staging path is always taken — the
  weights' pinning state is irrelevant, the staging buffer itself is pinned);
- deterministic garbage on every run (same copy order + same timing on the
  same GPU → same corruption; 1299 of ~1668 pool fills come from the racy
  prefetch path, only 369 from the correct miss-store path);
- early garbage output (cross-layer prediction fills the next layer's slots
  during prompt decode, so generation hits corrupted slots from the first
  token).

Aggravating: the code's *intended* async architecture (worker thread +
`fate_prefetcher::on_layer_done()` + `fate_prefetcher::submit()`) is dead code —
`on_layer_done()` and `sync()` are never called, so the worker thread never
receives jobs. Only the inline (racy) path runs.

## Fix (applied + tested)

Replace the single staging buffer with a small ring of pinned staging buffers,
one CUDA stream per slot, and `cudaStreamSynchronize` the slot's stream before
reusing it:

- `fate_prefetcher`: `N_STAGE = 16` buffers × `(expert_bytes_max + 512)`,
  each with its own stream (reuses the already-exported
  `fate_prefetch_stream_create/sync/h2d` primitives — no ggml-cuda.cu changes).
- In `on_expert_copy()`: pick ring slot `idx = prefetched % N_STAGE`,
  sync its stream (cheap: the previous DMA on that slot finished long ago),
  `memcpy` into it, enqueue the async H2D on that slot's stream.
- After the batch, insert the GPU barrier for **every** stage stream so the
  main compute stream waits for all of them.

No correctness regression; async overlap mostly preserved (ring of 4 makes the
per-copy sync effectively free).

Test result (build-cuda, same repro): **clean output restored**, stats:

```
======== FATE CACHE STATS ========
  accesses   : 18729
  hits       : 7935 (D2D from pool)
  misses     : 10794 (H2D fallback)
  hit rate   : 42.37%
  prefetched : 22146 (async H2D to pool)
  pool       : 2495 slots × 0.8MB
==================================
```

with correct text. (The broken run's 97.9% "hit rate" was the bug serving
garbage from the pool; the fixed run's misses are correct H2D fallback copies.)

## Files / evidence

- scratch/qwen-test-notes.md — full run notes
- scratch/fate_repro_cuda_broken.out, scratch/fate_repro_cuda_stage1_baseline.out
- (was) fate_diag*.out, fate_completion_test.out, fate_nommap.out — moved to scratch/
