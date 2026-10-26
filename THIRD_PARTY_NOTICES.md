# Third-party notices

FlintNSteel is licensed under Apache-2.0 (see [LICENSE](LICENSE)); project attribution is in [NOTICE](NOTICE).
The files below contain code or text copied or derived from third-party projects. Each file's
header names its upstream source, and the upstream license text is included here. Files under
`licenses/` are unmodified copies of the upstream license texts.

Derived code keeps the original copyright notice and license of its upstream project. Code that is
original to FlintNSteel is licensed under Apache-2.0 and is not listed here.

## Pinned upstream revisions

| Project | Pinned revision | License | Used for | License file |
|---|---|---|---|---|
| ggml / llama.cpp (ggml-org/llama.cpp) | `19ab5a5e65736a35437e6b1c0ccd6562fed01b1b` | MIT | ggml-cuda kernels included or transcribed (MMVQ, MMQ, FlashAttention, quantization), ggml-common block types | [licenses/MIT-llama.cpp-ggml.txt](licenses/MIT-llama.cpp-ggml.txt) |
| Strata (Niko1221/Strata) | `011c9e600a93bbc9fd451c563501b5831c32bbca` | MIT | CUDA caching pool, Philox sampler | [licenses/MIT-Strata.txt](licenses/MIT-Strata.txt) |
| rec (recursive-oss/rec) | `8af99abaf50b33903147d89eed91d1bb3ac642dd` | Apache-2.0 | chat template, message normalization, tool-call parsing, grammar builder | [LICENSE](LICENSE) |
| XGrammar (mlc-ai/xgrammar) | `5b4e9ce9e72524037ae24ecd831b9b6604d2eb48` (v0.2.1) | Apache-2.0 | linked C++ grammar library; structural tag translation | [licenses/Apache-2.0-xgrammar.txt](licenses/Apache-2.0-xgrammar.txt) |
| picojson (within XGrammar `3rdparty/`) | as shipped in the XGrammar revision above | BSD-2-Clause | JSON parsing | [licenses/BSD-2-Clause-picojson.txt](licenses/BSD-2-Clause-picojson.txt) |
| exllamav3 (turboderp-org/exllamav3) | v1.6.2 `4c842731`, Flint patch `3b013de9` | MIT | EXL3 kernels, routing and vendored CUDA headers | [licenses/MIT-exllamav3.txt](licenses/MIT-exllamav3.txt) |

The xgrammar library is not vendored into this repository. The build uses an external checkout
(see [docs/BUILD.md](docs/BUILD.md)).

## Files with copied or derived code

### ggml / llama.cpp (MIT)

Copyright (c) 2023-2026 The ggml authors. Full text: [licenses/MIT-llama.cpp-ggml.txt](licenses/MIT-llama.cpp-ggml.txt).

- `src/ggml/ggml_mmvq.cu` includes `ggml/src/ggml-cuda/mmvq.cu` unmodified.
- `src/ggml/ggml_mmq.cu` calls ggml's MMQ kernels and template instances unmodified.
- `src/ggml/ggml_fattn.cu` includes `fattn-mma-f16.cuh` unmodified.
- `src/ggml/mmq_table.cuh` is generated from `ggml/src/ggml-cuda/mmq.cuh` by `tools/gen_mmq_table.py`; its header carries the upstream copyright and the SHA-256 of the source file.
- `src/kernels.cu`: `block_q8_0` and the Q8_0 quantization in `store_q8_0` are transcribed from `ggml-cuda/cpy-utils.cuh`.
- `src/prefill_kernels.cu`: `block_q8_0` is transcribed from `ggml-common.h`.
- `src/engine.cu` and `src/prefill_kernels.cu` follow llama.cpp's graph and formulas; the comments at each site name the upstream file.

### Strata (MIT)

Copyright (c) 2026 Niko1221 and the Strata contributors. Full text: [licenses/MIT-Strata.txt](licenses/MIT-Strata.txt).

- `src/ggml/ggml_cuda_host.cu`: `CachingPool` is reduced from `src/prefill/ggml_cuda_host.cu`.
- `src/kernels.cu`: `philox_uniform` is taken from `src/kernels/cuda/sampler.cu`.

### rec (Apache-2.0)

Copyright and license: [LICENSE](LICENSE) (rec upstream). rec provides no NOTICE file.

- `serve/chat_template.jinja`: the same content as `engine/src/rec_mlx/chat_template.jinja`, with LF line endings. The template rendering is not modified.
- `serve/fns_server.py`: `normalize_messages` and `parse_tool_call` are derived from `engine/src/rec_mlx/chat.py`; the rest of the server is original.

### XGrammar (Apache-2.0)

Copyright (c) 2024 by XGrammar Contributors. Full text: [licenses/Apache-2.0-xgrammar.txt](licenses/Apache-2.0-xgrammar.txt). Attribution: [NOTICE](NOTICE).

- Linked, not vendored. `tools/fns_serve/fns_serve.cpp`, `tests/xgrammar_smoke.cpp`, and `tools/xgrammar/build.bat` use its public C++ API.
- `serve/flint_grammar.py` follows the structural tag of v0.2.1 as noted above.

### picojson (BSD-2-Clause)

Copyright 2009-2010 Cybozu Labs, Inc.; Copyright 2011-2014 Kazuho Oku. Full text: [licenses/BSD-2-Clause-picojson.txt](licenses/BSD-2-Clause-picojson.txt).

- Used unmodified from `3rdparty/picojson/picojson.h` of the XGrammar checkout, via `tools/fns_serve/fns_serve.cpp`, `tests/xgrammar_smoke.cpp`, and `src/exl3/exl3_model.cpp`.

### exllamav3 (MIT)

Copyright (c) 2025 Turboderp. Full text: [licenses/MIT-exllamav3.txt](licenses/MIT-exllamav3.txt).

- `src/exl3/vendor/` (CUDA headers and quantization kernels): kept as the upstream files, with the MIT header retained in each file. Interface changes are listed in each file header (for example, the removed Tensor interface in `exl3_moe_coop.cuh`).
- `src/exl3/exl3_kernels.cu` and `src/exl3/exl3_engine.cu`: dispatch and arithmetic follow exllamav3 files named in their headers, under the MIT notice in each file.
- EXL3 is experimental: its full-model parity is not validated and it is not recommended for serving.

### llama.cpp KL-divergence scoring (MIT)

Copyright (c) 2023-2026 The ggml authors. Full text: [licenses/MIT-llama.cpp-ggml.txt](licenses/MIT-llama.cpp-ggml.txt).

- `tools/fns_kld/kld_score.h`: the KL base reader and statistics are transcribed from `tools/perplexity/perplexity.cpp` (llama.cpp `19ab5a5e`, lines 191-239 and 1750-1758); the file header carries the MIT notice.


## Model weights and tokenizer

Model weights are not distributed with this repository. They are obtained separately and remain subject to their own license terms:

- Unsloth Dynamic GGUF weights for Flint (`unsloth/flint-GGUF`).
- The Flint tokenizer (`RecursiveSI/flint`, `tokenizer.json`), which is loaded at runtime and is subject to the upstream model license.

Refer to the upstream model cards for the exact license terms.

## SPEED-Bench preparation and evaluation data

The FNS-SPEED v1 preparation tools reference the official NVIDIA Model Optimizer resolver at
[`a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260`](https://github.com/NVIDIA/Model-Optimizer/tree/a4dfb916c8d4f958e11ff4dea8fd594f4ecc4260/examples/specdec_bench).
Its upstream code is Apache-2.0. Acquisition retrieves the pinned resolver and its upstream license into the
private work directory and records their provenance and digests; the public repository does not redistribute
the official resolver class or a wholesale transcription. `bench/scripts/speed_sources.py` is the public
acquisition adapter; the controlled loader reuses the original class locally with source acquisition
constrained to recorded revisions. The private resolver retains its upstream SPDX/Apache header verbatim,
and the upstream root `LICENSE` is fetched and hashed alongside it.
This code license does **not** license its datasets or remove their access conditions.

The dataset is [`nvidia/SPEED-Bench`](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/fbd6dc64353a3153bf24252a113ebd6d2cff9dec),
revision `fbd6dc64353a3153bf24252a113ebd6d2cff9dec`, under the
[NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/fbd6dc64353a3153bf24252a113ebd6d2cff9dec/License.pdf).
It is acquired separately, only after explicit acceptance, and is not included in this repository or release
assets. The agreement permits evaluation-result publication but prohibits dataset redistribution, including
portions and associated annotations/metadata, and prohibits training AI solutions with the data.

The resolver also uses third-party sources. Their individual licenses, source-specific terms and any account
or access gates remain applicable independently of both the NVIDIA dataset agreement and Apache-2.0.
Do not infer a blanket permissive license: HLE can be gated, OPUS100's card does not establish a known license,
ChatRAG delegates licensing to individual underlying sources (including SQA), and LiveCodeBench's broad CC card
label alone is not a complete source-terms review. Acquisition records exact source revisions where available,
file hashes, license records and resolver provenance locally, and fails if a required source is unavailable.
The official AdaLEval OpenCompass HTTP URLs are unversioned. The acquisition adapter uses the approved
third-party [`AmamiSora/LOOM_Data` mirror](https://huggingface.co/datasets/AmamiSora/LOOM_Data/tree/73c39cc596c1d69d009a07e41a3af57c819fbbb7/Ada_LEval),
revision `73c39cc596c1d69d009a07e41a3af57c819fbbb7`, to reconstruct the exact original JSON bytes from pinned
JSONL files. Both mirror and reconstructed-original SHA-256 digests are checked; record order and duplicate
occurrences are preserved. The official and mirror cards do not establish an explicit permissive dataset
license or redistribution grant. This technical reacquisition route grants no additional rights; source
review/consent remains required, and no dataset rows are included in the repository or release assets.

All source rows, prompts, generated outputs, token IDs, row identifiers, raw logs and per-example metadata
stay local. Only explicitly allowlisted, reviewed aggregate evaluation reports are candidates for publication.
See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for consent, reproducibility and disclosure rules.
