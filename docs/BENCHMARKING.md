# Benchmarking

## Protocol and scope

**FNS-SPEED v2 — adaptation of SPEED-Bench, single-turn, concurrency 1** is a paired HTTP performance
protocol for FlintNSteel and llama.cpp on Windows with an RTX 3090-class sm_86 GPU. It is **not an official
SPEED-Bench score**, NVIDIA certification, or a directly comparable entry in the paper's tables. Different
models, hardware, output budgets and concurrency invalidate such comparisons.

This guide describes the implementation and reproduction procedure, not measured results. No speedup or
completed acquisition, smoke test or release campaign is implied by the presence of the tooling.

The workload uses only the first user turn of each selected example. It does not evaluate answer quality,
pass@k, training, multi-turn KV reuse, tool execution or time to solve an agent task. Generation has a fixed
256-token budget with `ignore_eos=true`; it can continue past a natural stopping point. All generated tokens,
including reasoning and special tokens, count toward that budget. There is no system prompt, tool definition
or grammar. Ordinary inference defaults are unchanged.

Input and execution versions are deliberately separate. [`protocol-v1.json`](../bench/speed/protocol-v1.json)
is the **immutable acquisition/preparation definition**: keep its bytes, source-lock hashes, selection key,
budgets and reacquisition procedure unchanged. [`execution-v2.json`](../bench/speed/execution-v2.json) defines
the active persistent-block execution protocol. Its exact fields are `schema_version: 2`,
`name: "FNS-SPEED v2"`, `block_size: 32`, `block_order: "config-category-prepared-order"`,
`arm_order: "alternating-block-repetition"`, `server_lifetime: "block-arm"`,
`warmup: "once-per-session"` and `cache_prompt: false`. Release and full campaigns still require at least
five repetitions; smoke campaigns may use fewer. The input selection, token budget and parity gates are unchanged.
Existing valid prepared inputs can be reused with their exact hashes; execution v2 alone does not require
reacquisition or re-preparation. Recheck their provenance normally rather than changing the input definition.

## Pins and dependencies

| Component | Immutable reference |
|---|---|
| SPEED-Bench dataset | `nvidia/SPEED-Bench` at `fbd6dc64353a3153bf24252a113ebd6d2cff9dec` |
| Official resolver reference | `NVIDIA/Model-Optimizer` at `a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260`, `examples/specdec_bench/` |
| llama.cpp | `19ab5a5e65736a35437e6b1c0ccd6562fed01b1b` |
| Flint tokenizer | `RecursiveSI/flint` at `2a247be0c0aec895ccb0ac7ca92229ec6df37288` |
| GGUF source | `unsloth/flint-GGUF` at `20468edfc38ab43d6914cc88b08cbecb05d0fc62` |
| Input/preparation protocol (immutable) | [`bench/speed/protocol-v1.json`](../bench/speed/protocol-v1.json) |
| Execution protocol | [`bench/speed/execution-v2.json`](../bench/speed/execution-v2.json) |
| Python dependency lock | [`bench/speed/requirements.lock`](../bench/speed/requirements.lock) |

The expected model SHA-256 values are:

```text
UD-Q3_K_XL  91b8d85ab6ff7fb095d9a972f9afa6f3fa2b9acb642867a079f3e013ea3c350c
UD-Q4_K_XL  63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f
```

Dataset files, external source files, tokenizer, template, Q4 routing profile, dependency lock, both protocol
artifacts, executables and relevant DLLs are hashed as well. A binary hash alone is not a build recipe: retain
the exact FNS and llama.cpp commits, source modifications and the [build commands](BUILD.md). A dirty checkout must be
identified by its patch or source snapshot digest, not described as the clean upstream commit.

See the [paper](https://arxiv.org/abs/2604.09557),
[pinned dataset card](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/fbd6dc64353a3153bf24252a113ebd6d2cff9dec/README.md),
[pinned resolver](https://github.com/NVIDIA/Model-Optimizer/blob/a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260/examples/specdec_bench/specdec_bench/datasets/speed.py)
and [upstream preparation instructions](https://github.com/NVIDIA/Model-Optimizer/blob/a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260/examples/specdec_bench/README.md).
Cloning Model-Optimizer at that commit is not sufficient to lock the data: its dataset and some external
source loads are otherwise floating. FNS records the actual external revisions and file hashes before timing.

## License consent and storage boundary

Read the [NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/fbd6dc64353a3153bf24252a113ebd6d2cff9dec/License.pdf)
and every applicable source license before acquisition. The NVIDIA agreement permits publication of evaluation
results, prohibits redistribution of the dataset, including portions, and prohibits using it to train AI
solutions. Its restrictions cover annotations and metadata, not just prompt text. Third-party sources keep
their own terms; Apache-2.0 on resolver code does not license the data it retrieves. See also
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

`fetch` requires explicit consent. Installing dependencies, starting FNS, running ordinary tests or preparing
offline is not consent. Only the acquisition command accesses benchmark sources; do not arrange for it to run
as an installation hook or CI step. Do not upload or package its data with a release.

Consent does not bypass upstream access gates. For example, HLE access can require account approval;
the OPUS100 card does not establish a known license, ChatRAG delegates terms to its individual sources
(including the SQA source used by the resolver), and LiveCodeBench's card gives only a broad CC label.
Do not infer blanket permission from those labels. Retain and review the exact terms and access conditions
recorded for the pinned sources; if required access is denied, acquisition fails rather than silently skipping data.

Default private storage is `perso/bench/speed/`, covered by the repository's `/perso/` ignore rule. Keep HF
caches and preparation temporaries there or in an explicitly chosen private location. An external work directory
needs its own equivalent access controls and version-control exclusions. A `.gitignore` entry is not encryption.

| May be published after review | Local only |
|---|---|
| Scripts, pins, protocol, dependency lock and portable commands | Original/resolved Parquet, downloaded sources and caches |
| Aggregate measurements, coverage, exclusion and failure counts | Prompts, normalized messages, inputs, generated text and token IDs |
| Artifact digests, anonymized hardware facts and numeric options | Raw SSE, engine logs, native token logs and exception text |
| Selection algorithm and its digest | Per-example records, `question_id`, `src_id`, annotations and row manifests |

Even apparently harmless source identifiers and row metadata stay private. Each reader reconstructs the selection
under their own license acceptance. Reports use a positive field allowlist, not a blacklist over raw logs.
`report` writes to a local review directory and never publishes or uploads it. Review the exported files again
before sharing; report generation and permission to publish are separate decisions.

### Source licensing and reproducibility tiers

Treat these as separate questions, not one checkbox:

1. **Permission:** acquire the NVIDIA snapshot only after consent, and review each required source's terms.
   A technically accessible source is not automatically licensed for the intended use.
2. **Exact local replay:** retain all required files and their SHA-256 digests in `source-lock.json`. Offline
   preparation must reproduce canonical messages and token IDs without a network request.
3. **Independent reacquisition:** all required sources must also have immutable, retrievable revisions or
   content-addressed artifacts. A retained local file with a hash but no stable reacquisition path is not enough.

Use immutable revisions and file digests; HTTPS alone does not establish immutability. The official AdaLEval
URLs are mutable. Acquisition instead uses the explicitly approved third-party mirror
[`AmamiSora/LOOM_Data`](https://huggingface.co/datasets/AmamiSora/LOOM_Data/tree/73c39cc596c1d69d009a07e41a3af57c819fbbb7/Ada_LEval)
at revision `73c39cc596c1d69d009a07e41a3af57c819fbbb7`. Verify each pinned JSONL digest, retain record order,
then reconstruct original JSON with `json.dumps(rows, ensure_ascii=False, indent=4)`, UTF-8, no final newline.
The reconstructed digest must equal the pinned original AdaLEval digest before use. This is byte-exact
reacquisition, not permission to substitute a similar dataset. The upstream and mirror cards do not establish
a permissive redistribution grant; retain those limitations and do not redistribute benchmark data.
Missing sources, unresolved placeholders, duplicate SPEED-Bench row identifiers and unexpected schemas
remain errors. For external `AdaLEval/stackselect` only, preserve
the pinned official resolver's first matching occurrence (`iloc[0]`) in locked source-file order, including
conflicting duplicate source IDs; never deduplicate or reorder that file. The lock records this policy and
source/distinct-ID counts. This explicit compatibility exception does not remove the independent
reacquisition requirement. Never change source pins, bytes or selection after observing performance;
a changed lock defines a new campaign. Source-specific access approvals remain required.

`source-lock.json`'s `reproducible=true` describes immutable byte reacquisition, **not** a legal guarantee or
a finding that every source license is permissive. Operators remain responsible for their source-specific
review, consent and upstream access approvals. Local consent/review records do not override unknown or
delegated source terms.

## Acquire and prepare offline

Run PowerShell from the repository root. Build the pinned engines from a **Visual Studio 2022 x64 Developer
PowerShell/Native Tools shell**, then prepare the tokenizer/vocabulary using [BUILD.md](BUILD.md) and
[USAGE.md](USAGE.md#tokenizer). If multiple Visual Studio versions are installed, do not rely on `build.bat`'s
automatic latest-instance selection: initialize the VS 2022 environment explicitly before building. Use a
fresh build directory when changing toolchains; objects from another MSVC version must not be reused. For an
already mixed build directory, rebuild all targets with `cmake --build build --config Release --clean-first`
from the VS 2022 environment. Use local tokenizer and template files for this workflow; set `FNS_TOKENIZER`
so that preparation never falls back to a first-use tokenizer download.
The benchmark environment is separate from the ordinary serving/test environment. The locked environment
targets Python **3.14.7**, Windows x64, with `datasets==5.1.0`, `pandas==3.1.0`, `pyarrow==26.0.0`,
`tiktoken==0.14.0`, `numpy==2.5.3` and the full transitive package set in `requirements.lock` (including serving
tokenizer/template dependencies). Installing packages does not acquire SPEED-Bench or accept any dataset license.

```powershell
New-Item -ItemType Directory -Force perso/bench/speed | Out-Null
py -3.14 -m venv perso/bench/speed/venv
$SpeedPython = "perso/bench/speed/venv/Scripts/python.exe"
& $SpeedPython -m pip install -r bench/speed/requirements.lock
# Preserve the existing Hugging Face credential location before relocating data caches.
# This resolves only the path; it does not print or copy the credential.
if (-not $env:HF_TOKEN_PATH) {
    $env:HF_TOKEN_PATH = & $SpeedPython -c "from huggingface_hub.constants import HF_TOKEN_PATH; print(HF_TOKEN_PATH)"
}
$env:HF_HOME = [IO.Path]::GetFullPath("perso/bench/speed/hf-cache")
$env:HF_DATASETS_CACHE = [IO.Path]::GetFullPath("perso/bench/speed/hf-cache/datasets")
$env:TEMP = [IO.Path]::GetFullPath("perso/bench/speed/tmp")
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$env:FNS_TOKENIZER = [IO.Path]::GetFullPath("models/prof/flint-tokenizer.json")
```

The tool enforces every pinned package version and records the Python implementation/version. Offline
preparation requires the identical environment recorded during acquisition, including the Python patch version.
Changing the resolver adaptation or dependency lock requires a new private acquisition directory, not silently
mutating a previous lock. A dependency lock is a reproducibility input, not evidence that a campaign has passed.
CUDA/MSVC and native build requirements remain those in [BUILD.md](BUILD.md).

The preparation tool also confines HF, tiktoken and temporary-file caches to its private `--work-dir`
(`hf/`, `tiktoken/`, `tmp/`); it refuses an in-repository work directory outside `perso/`. The environment
settings above keep related manual acquisition work private too. They are not a way to bypass that boundary.
For gated sources, authenticate with your own Hugging Face account and obtain approval through the upstream
source's normal process. Keep `HF_TOKEN`/credential files out of campaign JSON and exported artifacts. The tool
preserves the existing `HF_TOKEN_PATH` when redirecting caches; it does not copy or log authentication tokens.

The first command below displays the license links and refuses before downloading because consent is absent.
After reading the NVIDIA agreement, the second command acquires the pinned NVIDIA snapshot, discovers the
required sources and refuses before third-party acquisition. Review the displayed source-specific URLs and
terms, obtain any required upstream account access, then explicitly give both consent tokens in the third
command. These flags are declarations by the operator, not evidence of a universal or permissive data license.

```powershell
& $SpeedPython bench/scripts/speed_prepare.py fetch `
    --protocol bench/speed/protocol-v1.json --work-dir perso/bench/speed

# Only after accepting the NVIDIA agreement:
& $SpeedPython bench/scripts/speed_prepare.py fetch `
    --protocol bench/speed/protocol-v1.json --work-dir perso/bench/speed `
    --accept-license nvidia-evaluation-dataset

# Only after reviewing and accepting the applicable third-party source terms as well:
& $SpeedPython bench/scripts/speed_prepare.py fetch `
    --protocol bench/speed/protocol-v1.json --work-dir perso/bench/speed `
    --accept-license nvidia-evaluation-dataset --accept-license third-party-sources

# No network: consumes only acquired, locked local sources and local tokenizer/template.
& $SpeedPython bench/scripts/speed_prepare.py prepare `
    --protocol bench/speed/protocol-v1.json --profile release --offline `
    --work-dir perso/bench/speed --tokenizer models/prof/flint-tokenizer.json `
    --template serve/chat_template.jinja
```

There is no implicit license acceptance and no offline fallback to an unpinned network load. `prepare` requires
`--offline`. `--profile smoke` or `--profile full` selects those distinct profiles; optional
`--configs CONFIG [CONFIG ...]` restricts those profiles, whereas `release` requires all six configurations.
Outputs are under `perso/bench/speed/prepared/PROFILE/`. Existing output may be reused only when its verified
content agrees; do not overwrite a prior campaign's selection. Repeating preparation against the same lock
must reproduce canonical messages, token IDs and order. Compare preparation digests before running engines.
Local JSONL records are delimited by physical newlines. Unicode line separators inside JSON strings remain
source content; they must not split records or change the resolved text before Flint normalization.

**Source-reproducibility boundary:** prior locks retaining mutable AdaLEval URLs remain unchanged and cannot
be promoted by editing `reproducible`. Create a new acquisition directory for the pinned mirror adaptation.
The same two explicit consent tokens apply, including review of the mirror and original source cards.
Only the ten pinned AdaLEval files are supported; unknown files, mirror-hash mismatches or reconstructed
original-hash mismatches fail. No arbitrary mirror or mutable-network fallback is accepted. Record the mirror
revision, downloaded digest, original digest and reconstruction recipe. Release/full still reject any lock
with non-reacquirable sources. Missing gated sources and unresolved placeholders remain fatal even for smoke.

## Selection and input equivalence

The six configurations are `qualitative`, `throughput_1k`, `throughput_2k`, `throughput_8k`,
`throughput_16k` and `throughput_32k`. Counts come from the acquired snapshot, not copied from the card.

For every row, take only nonempty `turns[0]` as the `user` message. Record unused later turns locally. Apply
FNS's existing message normalization and Flint chat template without rewriting, summarizing or translating
content. The prepared artifact records original/normalized text digests and transformations. Check normalization
idempotence. The fixed template determines reasoning and assistant generation prefixes.

Admissibility is checked **after** rendering and tokenization: `prompt_tokens + max_tokens <= ctx`.
The standard protocol is `ctx=32768`, `max_tokens=256`. No truncation, splitting or automatic output reduction
is permitted. `throughput_32k` is an upstream construction label, not a guarantee that the Flint prompt fits.

Within each configuration/category, sort by SHA-256 of compact UTF-8 JSON
`["fns-speed-v1", config, str(question_id)]`, breaking ties by `str(question_id)`. Selection does not depend on
Parquet row order, Python's process-randomized `hash()`, output content, acceptance rates or measured speed.

| Profile | Selection | Permitted interpretation |
|---|---|---|
| `smoke` | One admissible example per category in requested configurations | Technical validation; never a release result |
| `release` | All admissible qualitative rows; up to 32 admissible rows per throughput category; all six configurations | Standard FNS-SPEED v2 campaign, unchanged v1 input selection |
| `full` | All admissible rows in requested configurations | Separately named expanded-coverage campaign |

Coverage includes totals, admissible and selected counts, context exclusions, and actual Flint token-length
minimum/maximum/p50/p95 by category/configuration, including zero selections. An empty category prevents a
configuration-wide conclusion. A filtered adaptation must not be called exhaustive original SPEED-Bench.

Before timing, llama-server must use the same template (`--jinja --chat-template-file`); the harness checks
`/apply-template` and `/tokenize`, including BOS, special tokens and generation prefix. **Every input token ID
must match**, not merely the length. Both services receive identical normalized messages and generation
options. A rendering/tokenization discrepancy blocks the campaign.

## Comparison matrix

Run each cell separately for Q3 (`UD-Q3_K_XL`) and Q4 (`UD-Q4_K_XL`), using the identical SHA-256-identified
GGUF file in both arms on the same machine. A ratio between Q3 and Q4 is not an engine speedup.

| Comparison | Arm A | Arm B | What it measures |
|---|---|---|---|
| `service` | llama-server, no speculation | FNS `--spec 3 --spec-sample 2` | Published FNS configuration against that baseline service |
| `speculation` | FNS `--spec 0 --spec-sample 0` | FNS `--spec 3 --spec-sample 2` | Lookup/verifier contribution within FNS |

Main campaigns use greedy sampling (`temperature=0`) with penalties disabled. Optional `--mode sample`
campaigns are separate: temperature 1, top-p 0.95, top-k 20, presence penalty 0, seeds 0 and 7. Disable
additional llama.cpp sampling filters explicitly. The same seed does not imply the same random stream between
engines, so service comparisons do not require identical output tokens. The FNS speculation ablation does
require exact output-token parity per prompt and seed.

Keep `--kv-global q8_0`, `--fa-int8`, `--fa-swa` and `--attention-mma` off in the main matrix. Each bit-changing
variant requires a separate labeled table. EXL3 is outside this protocol. Freeze the Q4 routing profile from a
non-SPEED corpus, with its recipe and digest; do not tune it on measured prompts. Choose llama.cpp offload,
KV-cache and placement settings using a separate disclosed calibration workload and freeze them before timing.
Record observed memory and placement; do not deliberately handicap the baseline. Existing online expert-cache
adaptation may remain enabled when explicitly declared.

## Configure, check, run and report

Start from [`bench/speed/campaign.example.json`](../bench/speed/campaign.example.json), not a shell command
string. All artifact paths resolve relative to the campaign JSON file; `${NAME}` explicitly expands an
environment variable. Engine commands are argument arrays with controlled artifact placeholders, never
`shell=true` commands. Keep the real campaign private because it contains local paths and provenance.

Before `check`, fill every artifact's `{ "path": "...", "sha256": "..." }` entry using the actual file-byte
SHA-256. Include both models, binaries, service Python source, tokenizer, template, vocabulary, profile,
input protocol, execution protocol, source lock, dependency lock and all relevant DLLs. In PowerShell, obtain a
digest with `(Get-FileHash -Algorithm SHA256 <file>).Hash.ToLowerInvariant()`; do not hash a path string.
Record exact source commits and portable build recipes, plus the separate Q4 profile and llama.cpp calibration
recipes. Replace example machine descriptions and environmental limits with your own predeclared conditions.
The campaign uses `schema_version: 2`, retains `protocol` for the unchanged v1 input artifact and requires
`execution_protocol` for the v2 artifact, each with path and SHA-256. Remove the obsolete `pair_order` field;
arm order is determined by execution v2, not a configurable fixed AB override. The manifest records
`protocol_name: "FNS-SPEED v2"`, the exact `execution_protocol` object and `execution_protocol_sha256`;
public provenance includes `hashes.execution_protocol`. Explicit generation/sampling options, timeouts and
a hashed warmup remain required. Source pins, hashes, preparation budgets and selected input IDs are validated.

The example's `../../../` paths are intentionally correct **after copying it to**
`perso/bench/speed/campaign.json`, not for executing it in `bench/speed/`. Set `FNS_MODELS` and `FNS_TOKENIZER`
to absolute local paths; the example uses `${FNS_MODELS}` and `${FNS_TOKENIZER}` explicitly. Keep the
`environment` object free of unnecessary credentials. Populate the initially empty `dlls` array with
`path`/`sha256` objects for all runtime DLLs relevant to the builds. Ports 8091 (llama) and 8092 (FNS) must be
free; stop any ordinary FNS launcher before the campaign. The runner itself owns process startup and shutdown.
GPU-process isolation is strict by default: before startup it excludes other compute processes; while the
server is loaded it permits unowned graphics-only (`G`) contexts but rejects unowned `C` and `C+G` contexts.
This is not absolute GPU isolation. Set `gpu_process_policy` to
`{"mode":"strict","allowed_graphics_paths":[]}` (also the default when absent).

The opt-in WDDM desktop policy permits predeclared full executable paths with graphical (`G` or `C+G`)
contexts, including mixed compute/graphics contexts. Unlisted, pure-compute, unknown or unverifiable processes
remain errors; load, temperature and free-VRAM limits still apply. Use `mode: "wddm_desktop"` with a nonempty
list of exact absolute Windows executable paths observed and reviewed before the campaign. No wildcards,
basename-only matches or automatic additions. NVIDIA XML process types and the compute-process inventory must
agree; unknown/missing classifications fail closed. Keep the machine-specific list and observed exceptions
private. Public reports identify desktop-session measurements without exposing paths or PIDs. Establish the
policy and display/workload environment before measurement; no automatic process termination.

The `sources` entries need the checked-out commit, recoverable repository URL and build recipe. Dirty source
trees are snapshotted with patch/source-tree digests in private `raw/` artifacts. A recovery declaration is not
proof that an archive is accessible: an independent reviewer must retrieve the declared source archive and
verify that its SHA-256 matches the exact generated snapshot digest in the manifest. The runner does not upload
archives or perform network verification of recovery locations. Review any source-only archive before sharing
it, and never publish the raw private run directory. A clean, retrievable source commit is the simplest route.
Public reports retain typed dirty/source-tree/build-recipe digests and optional patch/snapshot digests, not
arbitrary private repository URLs or build command text. Retain the complete portable build recipe separately
with reviewed public source provenance.
For a dirty source entry, the exact configuration keys are `snapshot_repository` (recoverable source-archive
location) and `published_snapshot_sha256` (lowercase 64-hex digest). The declared digest must equal the runner's
generated `snapshot_sha256`; without both matching declarations the run is not publication-reproducible.
First obtain the generated source-only archive in a local run, review it, make it independently recoverable
and verify reacquisition, then create a separate final campaign configuration with those declarations.
Do not edit and resume the earlier campaign merely to relabel its provenance. A source change requires a new
archive digest and campaign; these declarations never authorize releasing licensed benchmark data.

The model-specific `fns_argv` and `llama_argv` arrays are explicit calibration inputs, not universally optimal
settings. In particular, review Q4 offload/cache choices against the declared independent calibration recipe.
The runner adds model/template/context/port/log/speculation arguments itself; do not duplicate or override
those reserved arguments in the arrays. Public exports reserialize recognized numeric/enum options and
replace artifact paths with symbolic slots; unknown options must acquire an explicit safe export rule rather
than being copied verbatim into a report.

```powershell
# Populate the copied configuration before checking; paths are relative to its destination.
Copy-Item bench/speed/campaign.example.json perso/bench/speed/campaign.json
$env:FNS_MODELS = [IO.Path]::GetFullPath("models")

# No performance campaign: validates local provenance and all selected prompt IDs on both llama quants.
& $SpeedPython bench/scripts/speed_bench.py check --campaign perso/bench/speed/campaign.json

# Main greedy matrix: four separate invocations, serially, never concurrently.
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q3 --comparison service
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q3 --comparison speculation
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q4 --comparison service
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q4 --comparison speculation

# Optional sampling matrix, still serial and separate from greedy results.
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q3 --comparison service --mode sample
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q3 --comparison speculation --mode sample
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q4 --comparison service --mode sample
& $SpeedPython bench/scripts/speed_bench.py run --campaign perso/bench/speed/campaign.json --quant q4 --comparison speculation --mode sample

# Use one completed run directory, or the collection root containing completed runs.
& $SpeedPython bench/scripts/speed_bench.py report `
    --run-dir perso/bench/speed/runs --out perso/bench/speed/public-review
```

`check` starts fingerprinted llama-server instances sequentially to validate template/tokenization; it is not
a no-GPU static lint and it does not run a latency campaign. Each `run` prints its unique directory under
the configured output root and rechecks input equivalence before measuring. Greedy is the default mode;
`--mode sample` uses seeds 0 and 7 declared by the protocol. The complete greedy plus optional sampling matrix
therefore has eight distinct runs, not a single blended score.

For a technical smoke, first prepare `--profile smoke --offline`, then point a separate private campaign's
`prepared_dir` at `prepared/smoke`, select a fresh output root and run the same four main cells. It may use
fewer repetitions, but its report remains NON-PUBLISHABLE. Do not replace the release selection with synthetic
native-smoke prompts or hand-picked high-acceptance examples.

To retry a failed or interrupted v2 run, keep its original campaign unchanged and pass the matching quant,
comparison and mode with `--resume <existing-run-directory>`. Resume is atomic at the **whole block pair**
for a seed/repetition: skip only when both latest arm sessions are complete and successful and every request
has valid latest rows in the same generation. Otherwise rerun all requests in both arms with a new block
generation, including previously successful rows inside that incomplete/failed unit. Preserve old rows,
session records and raw evidence, with explicit linked attempts; no invisible retry or cross-generation
cherry-picking. A completed session alone is insufficient if parity has invalidated its request rows.
Do not resume into a changed selection, budget or configuration. Use a new review directory per export.

There is **no v1 execution/resume/report compatibility**: start a new schema-v2 campaign and run root, do not
migrate or merge old attempts/sessions, and do not combine v1 and v2 results into one table. Historical code
and private artifacts remain preserved separately. Invoke `speed_bench.py` directly as above.

## Isolation and timing

The statistical pair remains `(quant, mode, configuration, category, example, seed, repetition)`; publication
still requires at least five pairs per example and seed. Execution groups requests by `(configuration,
category)` in first-occurrence order, then slices each group into blocks of **up to 32** in prepared order.
Block IDs start at zero. Loop over block, seed, then repetition (`pair`, zero-based). Use A then B when
`(block_id + pair) % 2 == 0`, otherwise B then A. Both arms traverse exactly the same request order. The two
block arms are adjacent, not each individual example's arms. Never overlap GPU servers or matrix cells.

For **each block arm, seed and repetition**, start one fresh server, wait for readiness, execute the fixed
out-of-corpus benchmark warmup **once per session**, then measure the whole block sequentially without restarting
or rewarming between requests. Engine startup may itself include an internal warmup. Finalize the session after
shutdown, including on failure, and release that server before starting the other arm. Record startup, warmup and
shutdown durations once in `sessions.jsonl`,
outside request latency; never attach those costs to the first/last request or repeat them on every row.

This is **KV-cold, weights-warm** measurement, not cold OS/hardware/expert caches. Loaded weights, allocations,
filesystem caches and declared online expert-cache adaptation can persist within a block. Warmup and earlier
requests can affect that state, which is why block membership/order and alternating arms are fixed and audited.
The measured request sets `cache_prompt=false`. FNS's ordinary default remains `true`. Explicit `false`
forces reuse to zero before prefill without removing the current prompt from lookup speculation's context.
Require native reused-token count zero and the full expected prefilled prompt length on **every** measured
request, including consecutive requests in the same session and the first request after warmup.

**Analytic process-count formula, not a measured gain:** for one quant/comparison/mode, let `n_g` be the selected
request count of each configuration/category group, `S` the number of seeds and `R` repetitions. Fresh
per-request execution required `2*S*R*sum(n_g)` measured-arm server sessions; v2 requires
`2*S*R*sum(ceil(n_g/32))` in a complete run without retries. These formulas exclude preflight tokenization
checks and retries; they establish neither wall-clock savings nor inference throughput. Failed whole-block
retries add sessions and requests.
Never use an invalid request as a cache-reset trick. Native metrics are correlated through a separate local
log using `X-Request-ID`, not timed console output. `--log-tokens` is opt-in, requires `--log`, and writes IDs
only to that private log; it does not add token IDs to the HTTP response.

Both arms use the same Python streaming client, loopback HTTP and a high-resolution monotonic clock:

| Measurement | Definition |
|---|---|
| Request latency | Just before sending the body through terminal SSE marker, with final usage received and validated |
| Visible TTFT | First nonempty `content` **or** `reasoning_content` delta; role-only deltas do not count |
| End-to-end tokens/s | Verified generated tokens divided by request duration in seconds |
| Indicative client TPOT | `(last nonempty delta time - first nonempty delta time) / (N - 1)` for `N > 1` |
| Streaming dispersion | Distribution of intervals between nonempty SSE deltas |

An SSE chunk is not a token. UTF-8 buffering, hidden special tokens, reasoning parsing and speculative emission
can group deltas. Never label SSE intervals per-token ITL or visible TTFT GPU prefill time. If there is no
visible delta, TTFT/TPOT are null with a reason, not zero; total latency can still be valid after native token
counts and termination are checked. Require `completion_tokens == max_tokens`; do not retokenize visible text
to reconstruct the count.

Internal engine timings are diagnostics with engine-specific boundaries, not directly interchangeable GPU
measurements. In FNS, `prefill_ms` includes first-token selection; `decode_ms` begins after that selection.
Post-first-token decode throughput is `(N - 1) / decode_seconds`, not `N / decode_seconds`.

### Native speculation counters

The native `done` record and correlated HTTP log retain `prefill_ms`, `decode_ms`, `spec_windows` and
`spec_drafted` and add:

- `spec_accepted`: draft tokens actually accepted **and emitted**, excluding correction/bonus tokens.
- `spec_emitted`: all tokens emitted by speculative verification calls.
- `decode_iterations`: decode calls after the first token, including ordinary no-draft steps and verification.

`spec_drafted` counts candidates submitted to verification, not lookup candidates rejected beforehand.
Drafts computed but not emitted because of the budget are not accepted; an emitted matching terminal draft
is accepted. Instrumentation must not change sampling choices or add a GPU synchronization to measured paths.

Report ratios from sums: `sum(spec_accepted)/sum(spec_drafted)`,
`sum(spec_emitted)/sum(spec_windows)` and `sum(N - 1)/sum(decode_iterations)`.
A zero denominator produces null with a reason. Never infer acceptance from `generated/spec_drafted`.
Require `0 <= spec_accepted <= spec_drafted`, `spec_emitted <= N - 1`, and zero speculative counters when
speculation is disabled.

## Statistics and failure gates

For each example, take each arm's median over its repetitions and declared seeds within the run, then compute
latency ratio `median(A)/median(B)`; a ratio greater than one favors B. Seed-specific pairing and exact FNS
ablation parity remain mandatory before this reduction. Also retain paired wins and dispersion. For each
configuration/category, report the median of per-example ratios, arm latency p50/p95, example count,
valid-pair count and paired-win fraction. Quantiles use nearest rank: sorted element `ceil(p*n)`, with
one-based indexing. Aggregate throughput is `sum(tokens)/sum(seconds)`, weighted by observed work and duration,
never a mean of request tokens/s.

Category tables are mandatory. Do not mix quantizations, modes, profiles or length configurations into a
single global multiplier. A configuration summary, when all categories are covered, may use the equal-category
geometric mean `exp(sum(log(category_median_ratio))/category_count)` and must state that formula.
A ratio below 1.02, or a win fraction below 80% (fewer than four of five pairs), is not called a stable gain. These
thresholds are a reporting convention, **not a statistical significance test**. Regressions remain visible.

Declare environmental invalidation limits before measurement. Record CPU/RAM, GPU/free VRAM, driver,
Windows/WDDM, CUDA/MSVC, build architecture, power plan, display state, PCIe link under load and engine memory
settings. Do not remove slow runs after seeing their latency.
The manifest's initial PCIe generation/width snapshot is taken **before** requests, not verified under load;
reports explicitly label it `pre_request_not_verified_under_load`. Separately, a read-only `nvidia-smi`
sampler runs at a nominal 200 ms interval, started before warmup and stopped after the block's requests, before server shutdown.
Only timestamped samples within the measured request window contribute to `gpu_during_request`; an active
sample has observed GPU utilization greater than zero. Reports aggregate safe category/arm sample counts,
active PCIe generation/width ranges and numeric memory/temperature/power/utilization observations. Missing or
zero-active observations have explicit null reasons, not invented loaded-link values. Sampling can miss short
events and is not a continuous hardware trace. The same monitoring runs for both arms, and its host-process
overhead is part of the declared campaign environment; it adds no CUDA synchronization or GPU setting change.
Telemetry uses host line-arrival timestamps and driver-averaged utilization, so delivery delay and averaging
prevent per-kernel timing claims. Release/full reports require a valid 200 ms sampler record for every
successful attempt and at least one active measured-window sample with PCIe generation/width ranges in every
selected category/arm; missing coverage rejects publication. Smoke may retain missing/zero-active observations
as null with an explicit reason. Reader/shutdown failures invalidate attempts as environmental errors.
Unsupported/nonnumeric rows remain private but are excluded from numeric summaries. Helper startup/shutdown
are outside request duration; concurrent monitoring overhead can still influence measured latency.

Before each block-arm session, inspect the full machine and require two seconds of stable idle state (250 ms
polling). Environmental rejection has a ten-second cooldown and a maximum thirty-minute wait; failure is
retained, not used to change the selected corpus. While the owned server is loaded, do not apply idle-GPU
utilization or free-VRAM startup thresholds to its own work. Session monitoring still enforces the declared
temperature ceiling (80 °C in the example campaign) and checks
unapproved GPU processes every two seconds, excluding owned job processes. It does not repeat a full system
fingerprint per request. The existing strict/explicit-WDDM policy and failure gates remain in force; no automatic
termination of outside processes.

Typed public provenance can include parsed CPU/GPU/Windows identity, architecture, driver mode, display state,
standard power-plan enum, DLL digests and Python version. Unknown or unsafe free-text observations become
null rather than being copied, so a null field is not evidence that the property was measured or absent.

Timeout, OOM, HTTP/stream error, incomplete output, unexpected KV reuse, absent metrics, environmental failure,
input-ID mismatch or FNS ablation-token divergence invalidates the affected work and stops the run. Preserve
attempts and raw errors locally. Speculation compares exact tokens between arms for each request; a mismatch
appends linked same-generation `spec_parity` invalidations, even when sessions had completed successfully.
There are no invisible retries; replacements explicitly link to the original attempts. A release report
requires all expected request/seed/repetition/arm rows to be valid within complete matching block generations,
verifies session membership and alternating order, and shows aggregated initial failures. An engine failure
is never a dataset exclusion. Smoke output is explicitly non-publishable; unit tests do not establish
performance or replace a real two-engine smoke.

## Local artifacts and public review

Each invocation creates a distinct run directory; do not append unrelated cells into one ambiguous file.

| Artifact | Purpose and disclosure |
|---|---|
| `source-lock.json` | Private acquisition lock: exact sources, revisions, digests, resolver and license records |
| `prepared/PROFILE/selection.jsonl` | Private eligibility/selection records, identifiers, lengths and transformations |
| `prepared/PROFILE/requests.jsonl` | Private normalized messages and prepared token IDs |
| `prepared/PROFILE/preparation.json` | Preparation digests, aggregate coverage and reproducibility state |
| `run-manifest.json` | Private schema-v2 commands/provenance, both protocols, machine/options and expected request set |
| `attempts.jsonl` | Private schema-v2 request timings/counters/status, output-token hashes, `block_id`, `block_generation`, `session_id`; linked attempts and global `observed_order` |
| `sessions.jsonl` | Private append-only schema-v2 block-arm lifecycle: complete request membership, generation/seed/repetition/arm, final status/environment/errors and startup/warmup/shutdown once |
| `raw/` | Private engine logs, SSE traces and token-parity evidence; generation-safe session identities prevent filename collisions on resume |
| `report.json`, `report.md` | Allowlisted aggregate export; review before publication |

Manifests distinguish exact file-byte hashes from canonical-content hashes. Canonical JSON is UTF-8,
`ensure_ascii=False`, sorted object keys, compact separators and no NaN. The source selection-key array retains
its specified order. Offline re-preparation must preserve canonical messages, IDs and ordering; Parquet byte
identity alone is not sufficient because serialization can change with libraries.

Request rows carry `pair_generation == block_generation` only to preserve pairing semantics, not v1 execution
compatibility. Session IDs are unique across generations; a successful block requires successful sessions and
all member rows to be successful. Request rows do not contain `startup_ms`, `warmup_ms` or `shutdown_ms`; session
`environment_before` is authoritative for admission. The public schema-v2 export exposes `runs[].sessions`
with `total`, `successful`, `failed`, `failed_by_code`, `unfinalized` and
`lifecycle_summed_ms: {startup, warmup, shutdown}`. Count each finalized session and its durations **once per
run**, including retained failed/replaced sessions, never again in category/arm latency tables. Interrupted
sessions without a finalized record are counted as `unfinalized`, without invented lifecycle durations.
The reporter authenticates the exact shipped execution-v2 file hash against manifest/public provenance.
Export only allowlisted aggregates, not session IDs, request membership, private paths, logs or raw errors.

The public export must contain no prompts, decodable output, per-example identifiers, personal paths, raw
exception strings or secrets. It should retain safe provenance, digests, declared options, selection rules,
budgets, coverage, aggregate failure reasons, statistical methods and limitations. Preserve full private
manifests for audit. Do not turn a denied release export into a claimed successful release by deleting failures.

See [TESTING.md](TESTING.md) for synthetic unit tests and native checks, and [USAGE.md](USAGE.md) for ordinary
serving behavior. The legacy `ab_fns_server.py` path includes a different warmup/workload, and `agent_replay.py`
requires a genuine agent trace: neither is a substitute for this protocol, and SPEED-Bench must not be wrapped
in a fabricated agent replay log.
