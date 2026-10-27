# Usage

This guide assumes the build in [BUILD.md](BUILD.md) succeeded. Commands are PowerShell, run from the repository root.

## Models

Put the GGUF files in `FNS_MODELS` (default `models\`). `bench/scripts/fetch-models.py` downloads them at the pinned
revision and verifies SHA-256:

| Key | File | Source |
|---|---|---|
| `q3` | `flint-UD-Q3_K_XL.gguf` (about 18.5 GB) | `unsloth/flint-GGUF`, revision `20468edfc38ab43d6914cc88b08cbecb05d0fc62` |
| `q4` | `flint-UD-Q4_K_XL.gguf` (about 25 GB) | same repository |
| `q8` | `flint-Q8_0.gguf` (about 42 GB) | same repository. Reference for KL measurements |

```powershell
py -3.14 bench\scripts\fetch-models.py q3 q4                 # destination: FNS_MODELS, else models\
py -3.14 bench\scripts\fetch-models.py --out <model-dir> q3  # explicit destination
```

With no key, the script fetches all three models. It refuses to start when the destination lacks free space for the
missing files. An existing file is not downloaded again, but its SHA-256 is still checked.

- **UD-Q3_K_XL** fits entirely in 24 GB VRAM with the default context.
- **UD-Q4_K_XL** is hybrid: experts that do not fit in VRAM are read from host memory.
  It needs the routing profile described in [TESTING.md](TESTING.md#q4-routing-profile).

## Tokenizer

The tokenizer is resolved in this order:

1. `FNS_TOKENIZER`, if set. A missing file is an error.
2. `FNS_MODELS\prof\flint-tokenizer.json`, if present.
3. The `RecursiveSI/flint` `tokenizer.json` from the Hugging Face hub, at revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288`.
   It is downloaded on first use and cached by `huggingface_hub`.

The chat server and the native engine use the same resolved file. The vocabulary for grammar constraints
(`FNS_MODELS\prof\flint-vocab.json`) is produced from that tokenizer:

```powershell
py -3.14 tools\xgrammar\export_vocab.py                # writes FNS_MODELS\prof (default)
py -3.14 tools\xgrammar\export_vocab.py <output-dir>   # explicit directory
```

The script writes `flint-tokenizer.json` (a copy of the resolved file) and `flint-vocab.json` (token strings indexed by id).
`fns_server.py` reads the vocabulary from `--vocab`, and `fns.py doctor` checks that it exists.

## Server

The OpenAI-compatible server is `serve/fns_server.py`. By default it starts `FNS_BUILD_DIR\fns-serve.exe`
(`build\fns-serve.exe` when unset) and exchanges JSON lines with it on stdin and stdout. It listens on `127.0.0.1` only.

```powershell
py -3.14 serve\fns_server.py --model models\flint-UD-Q3_K_XL.gguf --port 8080 --ctx 32768
```

Endpoints: `GET /v1/models` and `POST /v1/chat/completions` (`"stream": true` for server-sent events). Any other path
returns 404. The chat template is `serve/chat_template.jinja`. Tool calls are constrained by a grammar compiled by xgrammar,
and responses include `reasoning_content`. Requests are served one at a time.

Server options (`serve/fns_server.py`):

| Option | Default | Notes |
|---|---|---|
| `--model` | required | GGUF path |
| `--port` | `8080` | |
| `--ctx` | `32768` | Context length in tokens |
| `--ubatch` | `2048` | Prefill batch size |
| `--expert-profile` | none | Routing profile for the hybrid expert cache (`.route`) |
| `--rebalance` | `0` | Rebalance the expert cache every N tokens. Same output bits; `0` is static |
| `--spec` | `3` | Speculative verification window (prompt lookup). Same tokens as sequential decoding. `0` disables |
| `--spec-sample` | native auto: `min(2, --spec)` | Sampled-request verification width; explicit `0` disables, otherwise `2..--spec` |
| `--kv-global` | `f16` | `q8_0` is opt-in and changes the bits (see below) |
| `--fa-int8`, `--fa-swa`, `--attention-mma` | off | Opt-in flags that change the bits (see below) |
| `--engine` | `FNS_BUILD_DIR\fns-serve.exe` (default `build\fns-serve.exe`) | Engine binary, for A/B comparisons |
| `--vocab` | `FNS_MODELS\prof\flint-vocab.json` | Vocabulary for grammar constraints |
| `--log` | none | JSON-lines log, one line per request |
| `--log-tokens` | off | Requires `--log`; includes generated token IDs only in the private local log |
| `--record` | none | Records each request body as JSON lines, for `bench/scripts/agent_replay.py` |

`--spec` takes 2 to 8, and `0` disables it. `--spec-sample` may explicitly disable sampled speculation with `0`
or select a width from 2 through `--spec`. Omitting it preserves the native automatic setting `min(2, --spec)`.
For a fixed benchmark configuration, pass both values explicitly; do not change ordinary serving defaults.

### Request cache control and private diagnostics

Chat requests accept the strict JSON boolean `cache_prompt`, default `true`. With `false`, the native engine
forces reused prefix length to zero and prefills the full prompt, even after a previous warmup. This does not
remove the current prompt from speculative lookup. Strings such as `"false"`, integers and null are not booleans.
The normal `true` default retains existing prefix reuse.

An optional `X-Request-ID` header (`[A-Za-z0-9._:-]{1,128}`) correlates requests with `--log`; it is local log
metadata, not a new public response field. The completed-request JSONL includes prompt/completion counts,
`reused_prefix`, native finish reason, `prefill_ms`, `decode_ms`, `spec_windows`, `spec_drafted`,
`spec_accepted`, `spec_emitted` and `decode_iterations`. FNS prefill timing includes selection of the first
token; decode timing begins after that selection. Neither is visible HTTP TTFT.

Use `--log-tokens` only when exact output-token auditing is needed. It requires `--log` and adds `token_ids`
to that file, never to the HTTP response. The log is flushed before terminal SSE. Logs, recorded requests
and generated token IDs can contain sensitive or licensed data: keep them local and do not publish them.
For definitions of counters and the fixed-length cold-KV workload, see [BENCHMARKING.md](BENCHMARKING.md).

### Options that change output bits

These are opt-in. Each changes the numerical result, so outputs differ from the default. They are not
interchangeable with each other:

| Option | Scope | Incompatibilities |
|---|---|---|
| `--kv-global q8_0` | KV cache of global-attention layers, Q8_0 | Not with `--fa-int8` |
| `--fa-int8` | FlashAttention INT8 for global-layer prefill | Not with `--kv-global q8_0` |
| `--fa-swa` | FlashAttention for sliding-window prefill, visible tiles only | |
| `--attention-mma` | MMA attention, decode and verification | GGUF only; requires F16 global KV (not `q8_0`) |

Use the same supported option set when comparing tools. `fns-ref` and `fns-bench` accept the flags above;
`fns-kld` takes `--fa-int8 on` and `--fa-swa on` rather than bare flags. `fns-verify` supports
`--kv-global` and `--attention-mma`, but not `--fa-int8` or `--fa-swa`. `fns_server.py` passes its flags to the engine.

`serve/fns.py` (below) does not pass `--attention-mma`, `--fa-int8` or `--fa-swa`. Use `fns_server.py` for those.

## Command-line launcher

`serve/fns.py` wraps the checks and the server. Its default port is `8091`. It writes its state to
`FNS_STATE\server.json` (default `%LOCALAPPDATA%\FlintNSteel`), its server log to `FNS_STATE\server.log` and its request
log to `FNS_STATE\requests.jsonl`.

```powershell
py -3.14 serve\fns.py doctor                   # GPU, driver, VRAM, executables, models, Python, pi
py -3.14 serve\fns.py serve --quant q4         # fns_server.py detached on http://127.0.0.1:8091/v1
py -3.14 serve\fns.py status
py -3.14 serve\fns.py stop
py -3.14 serve\fns.py launch pi                # start the server if needed, then pi with an isolated config
```

Options:

- `--quant q3` (default) or `--quant q4`. The model is `FNS_MODELS\flint-UD-<quant>_K_XL.gguf`.
- `--port` (default `8091`), `--ctx` (default `32768`) and `--timeout` (server wait in seconds, default `300`) for `serve`
  and `launch`. `doctor` takes `--quant` and `--pi`.
- `launch pi` requires the `pi` client on `PATH` (or `--pi <path>`). Arguments after `--` go to `pi`.

`serve --quant q4` passes the routing profile `FNS_MODELS\prof\q4-code-train-8k.route` and `--rebalance 64`. It stops
with an error when the profile is missing. Create it as described in [TESTING.md](TESTING.md#q4-routing-profile).

`doctor` checks the NVIDIA GPU (compute capability 8.6), free VRAM (at least 20000 MiB, unless a FlintNSteel server is
already loaded), `fns-serve.exe` in `FNS_BUILD_DIR` (default `build\`), the selected model, the Q4 profile when
applicable, the vocabulary, the chat template, the Python packages, the tokenizer, and `pi`. It exits with status 1
when any check fails.

## Native tools

The engine binaries in `build\` have tool-specific options:

| Binary | Purpose |
|---|---|
| `fns-serve.exe` | Engine process driven by `fns_server.py` (JSON lines on stdin/stdout) |
| `fns-ref.exe` | Teacher forcing over a token file; `--profile-out` writes the routing profile |
| `fns-bench.exe` | Decode and prefill throughput; `-o jsonl` output |
| `fns-kld.exe` | KL divergence and top-1 agreement against a Q8_0 base (`--base`, `--chunks`) |
| `fns-verify.exe` | Verification at W columns: parity, causality, greedy speculative decoding |
| `fns-dump.exe` | Reference dump from the pinned llama.cpp |

Example, decode benchmark on the Q3 model:

```powershell
build\fns-bench.exe --model models\flint-UD-Q3_K_XL.gguf -n 256 -d 4096 -r 3 -o jsonl
```

KL against a Q8_0 base (`fns-kld` reads a llama.cpp `--kl-divergence-base` file that you produce separately):

```powershell
build\fns-kld.exe --model models\flint-UD-Q3_K_XL.gguf --base <path>\base.kld --chunks 32 -o kld.json
```

## Profiling and replay

These scripts need caller-supplied inputs. They have no default trace, model or session.

- `bench/scripts/prof_decode.py <name> --model <gguf> [fns-bench args...]` and
  `bench/scripts/prof_prefill.py <name> --model <gguf> [fns-bench args...]` run one `fns-bench` under Nsight Systems
  (`nsys` must be on `PATH`). They write `FNS_BUILD_DIR\bench\prof\fns-<name>.nsys-rep` and `.sqlite` (or `--out DIR`) and
  print the per-kernel breakdown. Example:

  ```powershell
  py -3.14 bench\scripts\prof_decode.py d0 --model models\flint-UD-Q3_K_XL.gguf -n 64 -d 0
  py -3.14 bench\scripts\prof_prefill.py pp32k --model models\flint-UD-Q3_K_XL.gguf -p 4096 -d 32768 -ub 2048
  ```

- `bench/scripts/agent_replay.py` replays a recorded session against llama-server and `fns_server.py` at equal work. It
  needs two files, both written by the server: `--trace` (request bodies, from `fns_server.py --record`) and
  `--trace-log` (the journal, from `fns_server.py --log`), plus `--model`. The llama.cpp arms need `llama-server.exe` from
  `FNS_LLAMA_BUILD`. Example, one fns_server option set against another:

  ```powershell
  py -3.14 bench\scripts\agent_replay.py --name q3-agent-spec --trace T.jsonl --trace-log T.log.jsonl `
      --model models\flint-UD-Q3_K_XL.gguf --fns "--spec 0" --fns-b "--spec 3" --out OUT_DIR
  ```

  Replay uses `ignore_eos`, so it disables the tool grammar. Use it only for fixed-length replay.
- `bench/scripts/route_profile.py` builds the Q4 routing profile and replays cache hit rates. See
  [TESTING.md](TESTING.md#q4-routing-profile).
- `bench/scripts/speed_prepare.py` and `bench/scripts/speed_bench.py` implement **FNS-SPEED v2 — adaptation of
  SPEED-Bench, single-turn, concurrency 1**. Acquisition requires explicit license consent; preparation remains
  offline under immutable `protocol-v1.json`, while `execution-v2.json` governs execution. A server handles
  up to 32 prepared requests per block arm, with one warmup per session and alternating A/B order between
  block repetitions. Weights stay warm; every measured request uses cold KV (`cache_prompt=false`) and full
  prefill. Session startup/warmup/shutdown are accounted once, outside request latency. Resume replaces an
  incomplete block pair in full, not individual successful rows. Do not resume or merge v1 execution artifacts.
  See [BENCHMARKING.md](BENCHMARKING.md) for the acquisition/check/run/report
  recipe, dependencies, privacy boundary and interpretation limits. This is not an agent trace or an official
  SPEED-Bench score.

Other scripts in `bench/scripts/` (`ab_*.py`, `kld.py`, `parity.py`, `exl3_dump.py`, `kld_exl3.py`) are not described
here; read each script's header and `--help` output before use. Benchmarks that compare against llama.cpp need the pinned
`llama-bench` and `llama-server` from `FNS_LLAMA_BUILD`.

## EXL3 weights (experimental)

EXL3 weights load from a safetensors model directory with `--weights exl3`. This path is **experimental and its
model-level parity is not validated.** It is not recommended for serving. `fns_server.py` does not use it.

```powershell
build\fns-bench.exe --weights exl3 --model <exl3-model-dir> -n 256 -d 4096 -r 3
```
