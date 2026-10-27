# Engine

This document describes the components, memory tiers, decode and prefill paths, and the options that change
numerical output. It states current limits and contains no benchmark results.

## Scope

- One model: Recursive Flint (39.8B MoE, about 2.5B active parameters per token, 256 experts with top-6 routing,
  49 layers, sliding-window and global attention).
- One device class: CUDA sm_86, Windows, one sequence (batch 1 for decoding).
- Weights: GGUF Unsloth Dynamic quantizations (`UD-Q3_K_XL`, `UD-Q4_K_XL`). EXL3 is experimental (see below).

## Components

| Component | Location | Role |
|---|---|---|
| `fns::Model` | `include/fns/model.h`, `src/model.cpp` | Maps a GGUF file; reads hyperparameters and tensor views. Throws `std::runtime_error` for a file that is not Flint |
| `fns::Engine` | `include/fns/engine.h`, `src/engine.cu` | Uploads weights, owns the KV cache, runs the decode, sampling, verification and prefill entry points |
| ggml kernels | `src/ggml/`, `include/fns/ggml_kernels.h` | Quantized GEMV (MMVQ), MMQ for prefill, Q8_1 quantization, expert sorting, FlashAttention; compiled from the pinned llama.cpp sources |
| Attention MMA | `src/attention_mma.cu` | Optional MMA attention for decode and verification (`attention_mma`) |
| INT8 FlashAttention | `src/fa_int8.cu` | Optional global-layer prefill attention (`fa_int8`) |
| Prompt lookup | `include/fns/lookup.h` | Draft tokens for speculative verification |
| `fns-serve` | `tools/fns_serve/fns_serve.cpp` | Process serving one engine over JSON lines; grammar masks through xgrammar |
| `fns_server.py` | `serve/fns_server.py` | OpenAI-compatible HTTP server; chat template, tool-call parsing, speculation settings |
| EXL3 | `src/exl3/` | Experimental weight path; see its own section |

## Memory tiers

- **VRAM**: dense and attention weights, KV cache, activation buffers, and the resident subset of experts.
- **Host memory**: the non-resident experts, which the GPU reads during decode. With EXL3, the token embedding is
  also in host memory.
- **Expert cache (Q4 hybrid)**: experts are placed in VRAM from a routing profile (`expert_profile`). With
  `rebalance_every` set, `maybe_rebalance` swaps experts between tokens. This changes placement, not output bits.

`Engine::placement()` returns an `ExpertPlacement` (bytes and expert counts in VRAM and in host memory).

Adaptive swaps use copies by default. `EngineOptions::rebalance_fused_swap` opts into an in-place GPU kernel exchanging
the same slot bytes through the mapped host address; `fns-bench --rebalance-swap fused` enables it and
`--rebalance-swap copy` selects the default. `fns-serve` and the Python server expose the same option
(`--rebalance-swap fused|copy`, default `copy`). This option applies to the GGUF engine.
`fns-swap-parity --model F.gguf` checks the exchange against the original bytes at the model's slot sizes and other sizes.

Set `FNS_REBALANCE_TIMING=1` before starting the engine to collect CUDA event time for swap and pointer-table work.
`Engine::placement()` exposes the accumulated milliseconds, calls with swaps, swaps, and logical bytes exchanged
(both slots per swap, excluding pointer tables); `fns-bench` prints these totals with its adaptive-cache line. The event
interval excludes routing-counter collection and includes any stream idle time between the first swap and final table
upload. Without the variable, timing events and timing counters are disabled.

## Layers

- Layers 0 and 1 are dense (FFN width 6144). The other layers are MoE.
- Sliding-window layers use RoPE and a window of 513 positions. The ring keeps `ring` positions (default 1024;
  at least 513), enlarged as needed to cover `513 + ubatch - 1` for prefill and
  `513 + verify_cols - 1` for verification.
- Global layers keep the whole context in the KV cache.

## Decode

Three entry points return the next token. They differ in where the choice is made and what the host receives:

| Call | Output to host | Choice |
|---|---|---|
| `step(token, pos, logits)` | Full logits, `[n_vocab]` floats | Host decides |
| `step_greedy(token, pos)` | Token id (4 bytes) | Greedy argmax on the GPU; ties choose the smallest index |
| `step_logits(token, pos)` + `pick(mask, sp, counter)` | Token id | Argmax, or top-k / top-p / temperature sampling on the GPU, restricted to a bitmask |

`pick` takes an optional xgrammar bitmask (`uint32 [(n_vocab + 31) / 32]`, 1 means allowed). It returns -1 when the mask
allows no token. `SampleParams` sets `temperature` (zero or less is greedy), `top_p`, `top_k` (0 means 64 candidates) and
the Philox `seed`.

`pick_async` / `step_next` / `pick_wait` (used by `fns-serve`) pipeline the decode. `pick_async` queues the choice and
its copy to the host. `step_next(pos)` queues the forward pass of the chosen token before the host reads the choice.
The host then waits once per token (`pick_wait`), while the next forward pass is already running on the GPU. If the
token is a stop token, that forward pass has already written its KV entry at `pos`, which stays valid.

`reset_presence()` clears the presence-penalty state at the start of each request.

## Prefill

`Engine::prefill(ids, n, p0)` processes `n` tokens at positions `p0 ..` in batches of at most `ubatch` tokens
(MMQ GEMM, FlashAttention). It fills the KV cache and returns the argmax of the last position's logits. It may also
copy each position's logits to the host for parity checks; this is slow and not used for serving. `pick` can be
called after `prefill` to sample the first generated token from the last logits.

## Verification and speculation

`Engine::verify(ids, W, p0, out, ...)` runs `W` consecutive positions in a single pass (`1 <= W <= verify_cols`).
It writes `out[c]`, the choice for column `c`, using the same selection rule as `pick`:

- Argmax when `sp.temperature <= 0`.
- Sampling with `SampleParams` and the Philox counter `counter + c` otherwise.
- Each column is restricted to its own bitmask `masks[c]` when masks are given.

The presence penalty counts `ids[1..c]` for column `c`. `presence_commit(n)` marks the first `n` choices as
committed after the caller accepts them.

Speculation in the server is prompt lookup: a draft of up to `W - 1` tokens is verified in one pass, and the
accepted prefix is kept. Because each column uses the same choice rule and the same counter as sequential decoding,
the emitted tokens are the same as sequential decoding for the same seed and sampling parameters. Rejected columns leave KV entries
that the next forward pass rewrites at the same position.

Speculation is set per binary:

| Binary | Default | Setting |
|---|---|---|
| `fns-serve` | disabled (`verify_cols = 0`) | `--spec W` for greedy windows (2 to 8); `--spec-sample W` for sampling (default `min(2, --spec)`; `0` disables it) |
| `serve/fns_server.py` | `--spec 3` | Passes `--spec` to `fns-serve` (`--spec 0` disables) |

The sampling window is smaller than the greedy window because sampled drafts are accepted less often.

## Options that change output bits

These options change the numerical result, so outputs differ from the default. Compare runs only with the same option set.

| Option | Effect | Incompatible with |
|---|---|---|
| `kv_global_q8` (`--kv-global q8_0`) | Global-layer KV cache stored as Q8_0 | `fa_int8`, `attention_mma` |
| `fa_int8` (`--fa-int8`) | INT8 FlashAttention for global-layer prefill | `kv_global_q8` |
| `fa_swa` (`--fa-swa`) | FlashAttention for sliding-window prefill, visible tiles only | None listed |
| `attention_mma` (`--attention-mma`) | MMA attention for decode and verification | GGUF only; requires F16 global KV, so `kv_global_q8` |

Defaults: F16 global KV, no INT8 FlashAttention, no SWA FlashAttention, no MMA attention.

Options that do not change bits: `rebalance_every` and `rebalance_fused_swap` (expert placement), and `verify_cols`
(speculation, which keeps the sequential result).

Quality is measured by `fns-kld` (KL divergence and top-1 agreement against a Q8_0 base) and `fns-verify`
(parity and causality). Quality statements need those tools.

## Limits

- Batch size 1 for interactive use. Multiple concurrent sequences are not supported.
- CUDA sm_86 only. No CPU backend, no Linux or macOS build.
- One loaded model per process.
- Context length is fixed at startup (`--ctx`, default 32768 in `fns-serve`).
- The EXL3 path is not validated at model level (see below).

## EXL3 (experimental)

`--weights exl3` loads an EXL3 safetensors directory instead of a GGUF file. Status:

- Isolated kernels (GEMV, GEMM, MoE, router, attention fixtures) match their reference.
- The full-model forward pass does not yet match ExLlamaV3. Model-level parity is **not validated**.
- `fns_server.py` does not use this path. The native protocol test does not show that
  `/v1/chat/completions` works with these weights.
- Prefill is limited to 8 tokens per batch; a larger `--ubatch` is capped in this path only.
- The BF16 embedding is read in zero-copy pinned memory; projections stay in VRAM.

**Do not use EXL3 to serve an agent.** Use the GGUF path.

## Kernel sources

The ggml kernels come from the pinned llama.cpp sources. The EXL3 kernels and vendored helpers come from exllamav3 at
a pinned revision, with local modifications listed in each file's header. Licenses and provenance are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
