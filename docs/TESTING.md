# Testing

This document lists the test programs, how to run them, and what they do and do not cover.

**No GPU CI.** The repository's CI (`.github/workflows/checks.yml`) is CPU-only. It runs on `windows-latest` with
Python 3.14, without CUDA, a GPU, models or the Hugging Face download. It runs the CPU unit tests and byte-compiles the
public Python files. No CUDA kernel, GPU test or model-level test runs in CI. The GPU programs below run only on a
machine with an sm_86 GPU and the models. Their presence does not imply any result. Run them yourself and record the output.

## Prerequisites

- For native test programs: a successful build ([BUILD.md](BUILD.md)). Binaries are in `build\` (or `FNS_BUILD_DIR`).
- The Python packages from `requirements-test.txt` ([BUILD.md](BUILD.md#4-python-environment)).
- For model-based tests: the GGUF model files in `FNS_MODELS` (see [USAGE.md](USAGE.md#models)).
- For `kernel_parity.py` and `route_profile.py hits`: the llama.cpp checkout at `FNS_LLAMA_DIR`, for its `gguf-py` package.
- For `fns-xgrammar-smoke`: the tokenizer and vocabulary `FNS_MODELS\prof\flint-vocab.json`
  ([USAGE.md](USAGE.md#tokenizer)). The CPU unit tests use a small in-memory tokenizer instead.

## CPU tests

The CPU unit tests need no GPU, no model and no network after the tokenizer package is installed:

```powershell
py -3.14 -m unittest discover -s tests -p "test_*.py" -v
```

`tests/test_service.py` covers message normalization, tool-call parsing, SSE splitting and incremental prompt encoding
with a small local tokenizer. `tests/test_assets.py` covers vocabulary export, repeated setup and explicit missing paths.

CI also byte-compiles the public Python files (`git ls-files '*.py'`, excluding `perso/`) with `python -m py_compile`.

### FNS-SPEED synthetic tests

The benchmark tests use invented fixtures, never excerpts or row identifiers from SPEED-Bench. They do not
acquire source data, imply license consent or measure throughput. Essential pure-core tests run with
`requirements-test.txt`, without importing the acquisition-only `datasets`/`pandas`/`pyarrow`/`tiktoken`
stack or contacting the network. Run the focused suites from the repository root:

```powershell
py -3.14 -m unittest discover -s tests -p "test_speed_prepare.py" -v
py -3.14 -m unittest discover -s tests -p "test_speed_bench.py" -v
py -3.14 -m unittest discover -s tests -p "test_speed_report.py" -v
py -3.14 -m unittest discover -s tests -p "test_service.py" -v
```

The suites cover deterministic selection and context eligibility, source/schema rejection, fragmented SSE
and usage validation, cache-control/logging contracts, statistics and allowlisted exports. Execution-v2
behavioral coverage must exercise deterministic blocks of up to 32 requests, alternating block/repetition
arm order, one startup/warmup/shutdown per block-arm session, full cold-KV prefill on every request, and
request-window telemetry from a session-lifetime sampler. Resume cases require whole-block-pair replacement
after failure/interruption, linked attempt chains, generation-safe session membership, retained failures and
speculation-parity invalidations. Reporting must reject incomplete blocks, mixed generations, incorrect
order/membership and v1 execution artifacts, and count session overheads once rather than per request.
These are lifecycle/behavior checks, not assertions about source-code text. The immutable v1 input protocol
and source-lock/reacquisition tests remain applicable; execution schema v2 is a separate contract. Native
cache behavior, GPU counter semantics and exact generated-token parity require the separate model-based
check below; synthetic tests cannot establish them or any performance gain.

## FNS-SPEED native contract smoke

This is an opt-in GPU test using locally authored synthetic prompts, **not** acquired SPEED-Bench rows.
It starts native processes sequentially for both quants and never downloads missing assets. Supply an existing
tokenizer and routing profile; the output directory must be new:

```powershell
py -3.14 tests/speed_native_smoke.py --engine build/fns-serve.exe `
    --q3 models/flint-UD-Q3_K_XL.gguf --q4 models/flint-UD-Q4_K_XL.gguf `
    --expert-profile models/prof/q4-code-train-8k.route `
    --tokenizer models/prof/flint-tokenizer.json --ctx 4096 --max-tokens 64 `
    --out-dir perso/bench/speed/native-smoke
```

Defaults are `--ctx 4096`, `--ubatch 2048`, `--max-tokens 64` and `--timeout 600` seconds. The test exercises
strict cache booleans, cold prefill after warmup, cached and omitted-default requests, and exact sequential/
speculative output-token parity in greedy and sampling modes (seeds 0 and 7). It reconstructs speculative
counters from the actual emitted IDs. Parity is compared at matching cache states, not between distinct
cold/cached prefill geometries. It requires observed no-draft, full-acceptance and in-window-budget coverage;
rare rejection/stop boundaries not exercised by the model are listed explicitly in `summary.json`, not claimed
as covered. Raw requests, tokens and engine diagnostics remain private in the output directory.

This check does not replace the real licensed-data smoke or release campaign. For those, follow
[BENCHMARKING.md](BENCHMARKING.md): acquire and resolve authorized sources, prepare `--profile smoke --offline`,
then check and run both quants and comparisons with a schema-v2 campaign and a fresh run root. Verify that
each measured request has zero reused tokens even when following another request in the same session, that
warmup occurs only once per block arm, and that session overheads are separate from request timings. A valid
real smoke must observe both speculation and no-draft paths; its report is always non-publishable as a release
score. Preserve old v1 evidence separately; it is not v2 verification and must not be resumed or merged.
Use the runner directly. Neither recipe states that it has been run.

## Live HTTP smoke test

`tests/http_smoke.py` checks a running OpenAI-compatible server. It sends requests to it and does not start one:

```powershell
py -3.14 serve\fns_server.py --model models\flint-UD-Q3_K_XL.gguf --port 8080
py -3.14 tests\http_smoke.py --url http://127.0.0.1:8080/v1
```

Options:

- `--url` (default `http://127.0.0.1:8080/v1`): base URL of the server, including `/v1`.
- `--timeout` (default `600`): per-request timeout in seconds.
- `--server-log <path>`: the JSON-lines file written by `fns_server.py --log`. With it, the disconnect check also verifies
  that the server logged the cancelled request and that it stopped before its token budget.

The script checks response structure and usage, cached-to-cached output equality, SSE framing, a `read_file` tool call
(non-streamed and streamed), and a client disconnect followed by a new request. It exits with status 0 only when every check
passes. Free-text output is never compared with a fixed string. Run it on the GPU machine; it is not part of CI.

## Test programs

Parity test executables fail on a mismatch. `fns-kernel-parity` only writes outputs: use `kernel_parity.py` to compare
them. `fns-verify` reports comparisons without failing on a mismatch; inspect its JSON for zero logit/KL differences,
`argmax_same == cols`, `causal_same == causal_checked`, and `match == n_gen` for each decode run.

| Program | Command | What it checks |
|---|---|---|
| `fns-kernel-parity` | `build\fns-kernel-parity.exe --model <gguf> --x <x.f32> --out <dir>` | Quantized GEMV outputs for one tensor per format, written as raw `.f32` |
| `kernel_parity.py` | `py -3.14 tests\kernel_parity.py [--model <gguf>]` | Runs `fns-kernel-parity` and compares with a float64 reference from `gguf-py`; writes `kernel_parity.json` in `FNS_BUILD_DIR` |
| `fns-expert-parity` | `build\fns-expert-parity.exe --model <gguf>` | Expert pointer tables (VRAM and mapped host memory) against ggml's `mmvq_ids`, per MoE layer |
| `fns-mmq-parity` | `build\fns-mmq-parity.exe --model <gguf> [--tokens 77]` | Prefill MMQ: expert tables and dense tensors against MMVQ and ggml's MMQ |
| `fns-attention-mma-parity` | `build\fns-attention-mma-parity.exe <gguf> <tokens.txt>` | MMA attention against ggml FlashAttention and sequential decoding; the tokens file needs at least 1024 ids |
| `fns-sample-check` | `build\fns-sample-check.exe` | Sampling frequencies against the top-k / top-p / temperature distribution. Takes no arguments |
| `fns-mmvq-bench` | `build\fns-mmvq-bench.exe --model <gguf> [--layer 3] [--reps 200]` | Timing of expert and dense GEMV kernels. A benchmark, not a pass/fail test |
| `fns-xgrammar-smoke` | `build\fns-xgrammar-smoke.exe --tokenizer <tokenizer.json> --vocab <vocab.json>` | xgrammar matcher against the Flint vocabulary (`tests/xgrammar_smoke.cpp`). CMake target `fns-xgrammar-smoke` |
| `fns-verify` | `build\fns-verify.exe --model <gguf> (--prompt <ids.txt> \| -d <D>) -W 2,4,8 [-n 256] [--draft oracle\|lookup\|both] [-o out.json]` | Verification at W columns: parity with sequential decoding, causality, greedy speculative decoding |

Run the GPU tests one at a time. Each loads a model and needs most of the VRAM.

`fns-xgrammar-smoke` needs no GPU. It needs the tokenizer and the vocabulary from [USAGE.md](USAGE.md#tokenizer). The
test binaries are built by the CMake targets in `CMakeLists.txt`; `fns-xgrammar-smoke` links only xgrammar.

### EXL3 parity (experimental)

`fns-exl3-parity` (source `tests/exl3_parity.cu`, program name `exl3-parity`) has three forms. None reads a default
model or data path. A required flag that is missing is an error.

```powershell
# 1. Fixture parity: linear, router and MoE against <parity-dir>/manifest.json. --model and --data are required.
build\fns-exl3-parity.exe --model <exl3-model> --data <parity-dir>

# 2. Attention fixtures only. Loads no model.
build\fns-exl3-parity.exe --attention-data <attention-dir>

# 3. Diagnostic trace of the forward pass. --model and --trace-out are required; --trace-out must not exist.
#    --trace-tokens is a raw int32 file.
build\fns-exl3-parity.exe --model <exl3-model> --trace-tokens <tokens.i32> --trace-out <new-dir> `
    [--batch 1|8] [--trace-start N] [--trace-count N] [--trace-layer L] [--replay <dir>] `
    [--reprepare-router] [--public-forward] [--decode] [--lifecycle] [--ring N]
```

EXL3 status: isolated kernels match their reference. The full-model forward pass does not yet match ExLlamaV3, so
these tests do not validate model-level behavior. A passing fixture run is not validation of the EXL3 model.

The EXL3 reference dumps need an exllamav3 checkout: `bench/scripts/exl3_dump.py` and `bench/scripts/kld_exl3.py` read
`FNS_EXLLAMAV3_DIR` or take `--exllamav3-dir`. Their output is not a validation of the EXL3 model either.

## Q4 routing profile

The Q4 model uses an expert-cache routing profile, `q4-code-train-8k.route`, in `FNS_MODELS\prof\`. The launcher
(`serve/fns.py serve --quant q4`) passes it to the server. If the profile is missing, the launcher stops with an error;
there is no silent fallback to another profile.

To regenerate it, run `route_profile.py profile`. Its inputs are:

1. The GGUF model `flint-UD-Q4_K_XL.gguf` (`--model`, default `FNS_MODELS\flint-UD-Q4_K_XL.gguf`).
2. A UTF-8 code corpus (`--corpus`, default `FNS_MODELS\kl\code.txt`). See [Corpus](#corpus).
3. `fns-ref.exe` in `FNS_BUILD_DIR`.
4. The tokenizer: `FNS_TOKENIZER`, else `FNS_MODELS\prof\flint-tokenizer.json`, else the pinned Hugging Face revision.

```powershell
py -3.14 bench\scripts\route_profile.py profile --model models\flint-UD-Q4_K_XL.gguf --corpus <corpus.txt> `
    --start 16384 --count 8192 --out models\prof\q4-code-train-8k.route
```

The default slice is tokens 16384 to 24575 (`--start 16384 --count 8192`). The corpus must contain at least
`--start + --count` tokens. Choose a disjoint corpus slice when generating the separate test profile for `hits`.

`route_profile.py hits --profile <p.route> --test <t.route>` replays the placement and reports the share of selections
served from VRAM. It does not need the GPU. Its `--host-bytes` option sets the expert bytes kept in host memory (default
2.13e9).

## Corpus

The quality and profile corpus is code text, which is third-party and not stored in this repository. It is rebuilt
from the pinned commits listed in `bench/corpus/code.manifest.json`:

```powershell
py -3.14 bench\corpus\build.py <output.txt>
```

`build.py` reads the source repositories from the checkouts named by environment variables, at the manifest commits:

- `FNS_LLAMA_DIR`: llama.cpp (default `external\llama.cpp`)
- `FNS_REC_DIR`: rec (required; the manifest references it)
- `FNS_STRATA_DIR`: Strata (required; the manifest references it)

It checks the SHA-256 of the result against the manifest (139526 tokens, `773512a1...`). Without an output argument it writes
`FNS_MODELS\kl\code.txt`.

Without the corpus, `route_profile.py profile` and the quality measurements in `fns-kld` cannot be reproduced.

## Optional xgrammar reference

`requirements-test.txt` names an optional, isolated xgrammar Python reference: `xgrammar==0.2.1` in a separate environment.
It is not installed by default. xgrammar's Python import loads torch, so keep it out of the serving environment. The
FlintNSteel server and tests do not need it. The grammar that the server uses is compiled by the native `fns-serve`.

## Scope and limits

- The tests check numerical parity with references and internal consistency. They do not measure task quality.
- CI covers only the CPU unit tests and byte-compilation (see the top of this document). It does not cover CUDA kernels,
  the GPU, the native test programs, or model inference.
- Parity tests use the models in `FNS_MODELS`; results depend on the exact GGUF files. The expected SHA-256
  values are in `bench/scripts/fetch-models.py`.
