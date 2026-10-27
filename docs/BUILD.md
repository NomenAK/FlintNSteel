# Build

This guide covers the prerequisites, the dependency pins, the variables that the build and tools read, and the build steps.

Target: Windows 10/11 x64, NVIDIA GPU with compute capability 8.6 (sm_86), CUDA Toolkit 13.4 (the tested version),
Visual Studio 2022 MSVC toolchain, Ninja.

## Prerequisites

| Tool | Requirement |
|---|---|
| Visual Studio 2022 | MSVC x64 toolchain ("Desktop development with C++"). `tools\build-env.bat` loads it through vswhere unless a developer shell is already active |
| CUDA Toolkit 13.4 | `CUDA_PATH` set by the installer. `nvcc` is taken from `CUDA_PATH\bin`, then from `PATH` |
| CMake 3.28 or newer | On `PATH` |
| Ninja | On `PATH`, or the full path in `FNS_NINJA` |
| Git | On `PATH` |
| Python 3.14 | Started with `py -3.14` |

Network access is needed for the clones in sections 1 and 2, for the model downloads ([USAGE.md](USAGE.md#models)), and
for the tokenizer download on first use when no local copy exists ([USAGE.md](USAGE.md#tokenizer)).

## Dependency pins

| Dependency | Pin | Used for |
|---|---|---|
| llama.cpp | `19ab5a5e65736a35437e6b1c0ccd6562fed01b1b` | ggml CUDA kernels (MMVQ, MMQ, Q8_1 quantization, MUL_MAT_ID sorting, FlashAttention), reference binary `fns-dump`, `gguf-py` for kernel tests and the routing profile |
| xgrammar | `5b4e9ce9e72524037ae24ecd831b9b6604d2eb48` (v0.2.1) | Tool-call grammar in `fns-serve` (required: the build fails without it); also provides `picojson` and `dlpack` headers |
| Tokenizer | Hugging Face `RecursiveSI/flint`, revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288` | `tokenizer.json` for chat, tokenization and the grammar vocabulary |
| GGUF weights | Hugging Face `unsloth/flint-GGUF`, revision `20468edfc38ab43d6914cc88b08cbecb05d0fc62` | `bench/scripts/fetch-models.py` downloads and checks SHA-256 |
| exllamav3 | Checkout, default `external/exllamav3` for `tools/vendor_exl3.py` | EXL3 experimental path only (not needed for GGUF). The vendored files in `src/exl3/vendor/` record exllamav3 v1.6.2 (`4c842731`) in their headers |

Vendored code keeps its own license headers and provenance comments; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Use these exact commits. Other versions are not tested.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `FNS_MODELS` | `<repo>\models` | Model directory: GGUF files, `prof\` (profile, vocabulary, tokenizer), `kl\` (corpus) |
| `FNS_LLAMA_DIR` | `<repo>\external\llama.cpp` | llama.cpp checkout (sources and `gguf-py`); CMake cache variable |
| `FNS_LLAMA_BUILD` | `${FNS_LLAMA_DIR}\build-sm86` | llama.cpp Release build directory (libraries and DLLs); CMake cache variable |
| `FNS_XGRAMMAR_DIR` | `<repo>\external\xgrammar` | xgrammar checkout; CMake cache variable |
| `FNS_XGRAMMAR_BUILD` | `${FNS_XGRAMMAR_DIR}\build-fns` | xgrammar build directory containing `xgrammar.lib`; CMake cache variable |
| `FNS_EXLLAMAV3_DIR` | `<repo>\external\exllamav3` in `tools/vendor_exl3.py`; none in the EXL3 scripts | exllamav3 checkout. `bench/scripts/exl3_dump.py` and `kld_exl3.py` require it or `--exllamav3-dir` |
| `FNS_BUILD_DIR` | `<repo>\build` | FlintNSteel build directory, read by `build.bat`, the Python tools and the tests |
| `FNS_NINJA` | none | Full path to `ninja.exe`, used when Ninja is not on `PATH` |
| `FNS_TOKENIZER` | see [USAGE.md](USAGE.md#tokenizer) | Explicit `tokenizer.json`. A missing file is an error |
| `FNS_STATE` | `%LOCALAPPDATA%\FlintNSteel` | State and logs of `serve/fns.py` |
| `FNS_REC_DIR`, `FNS_STRATA_DIR` | none | rec and Strata checkouts, read by `bench/corpus/build.py` ([TESTING.md](TESTING.md#corpus)) |
| `CUDA_PATH` | set by the CUDA installer | CUDA Toolkit root. `nvcc` is taken from `CUDA_PATH\bin`, then from `PATH` |

Every CMake variable can also be set with `-D`, for example `-DFNS_LLAMA_DIR=<path>`.

No runtime path depends on a personal or machine-specific location.

## 1. Build llama.cpp

Clone and pin the checkout, then build it with the reference script:

```powershell
git clone https://github.com/ggml-org/llama.cpp external\llama.cpp
git -C external\llama.cpp checkout 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b
bench\scripts\build-llama.bat
```

`bench\scripts\build-llama.bat` builds Release for CUDA sm_86 (`86-real`) with flash attention for the F16 and q8_0 KV
pairs, and builds the targets `llama-bench`, `llama-server`, `llama-perplexity` and `llama-cli` in
`external\llama.cpp\build-sm86`. It takes extra `-D` options as arguments.

The FlintNSteel configure step requires `ggml-base.lib`, `ggml.lib`, `llama.lib` and the `ggml-base`, `ggml`, `ggml-cpu`,
`ggml-cuda` and `llama` DLLs under `FNS_LLAMA_BUILD`. It stops with the name of any missing file. If the checkout
lives elsewhere, set `FNS_LLAMA_DIR` (and `FNS_LLAMA_BUILD` if the build directory differs) before the build.

## 2. Build xgrammar

```powershell
git clone --recurse-submodules https://github.com/mlc-ai/xgrammar external\xgrammar
git -C external\xgrammar checkout 5b4e9ce9e72524037ae24ecd831b9b6604d2eb48
git -C external\xgrammar submodule update --init --recursive
tools\xgrammar\build.bat
```

Update the submodules after the checkout, so they match the pinned commit. `tools\xgrammar\build.bat` writes its own
`config.cmake` into `FNS_XGRAMMAR_BUILD` (Python bindings, C++ tests and coverage off), configures and builds the library,
and checks that `xgrammar.lib` exists. It warns when the checkout HEAD is not the pin. It writes only to the build directory
(`FNS_XGRAMMAR_BUILD`, which by default lies inside the checkout).

xgrammar is required. The FlintNSteel configure step fails when `xgrammar.h`, the `picojson` and `dlpack` headers, or
`xgrammar.lib` are missing, and `fns-serve` always links it.

## 3. Build FlintNSteel

The supported way is `build.bat` at the repository root. It loads the MSVC x64 toolchain when needed, configures
Ninja, Release, `86-real` into `FNS_BUILD_DIR` (default `build\`), and builds. Extra arguments are passed to the configure
step as `-D` options:

```powershell
.\build.bat
```

Equivalent manual steps, from a Developer PowerShell for VS 2022 in the repository root, with `cmake`, `ninja` and `nvcc`
on `PATH`:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real `
  -DFNS_LLAMA_DIR=external/llama.cpp -DFNS_LLAMA_BUILD=external/llama.cpp/build-sm86 `
  -DFNS_XGRAMMAR_DIR=external/xgrammar -DFNS_XGRAMMAR_BUILD=external/xgrammar/build-fns
cmake --build build --config Release
```

The build produces, in `build\`:

- `fns-serve.exe`: engine process driven by `serve/fns_server.py` and `serve/fns.py`
- `fns-ref.exe`, `fns-bench.exe`, `fns-kld.exe`, `fns-verify.exe`, `fns-dump.exe`: the native tools in [USAGE.md](USAGE.md#native-tools)
- `fns-kernel-parity.exe`, `fns-expert-parity.exe`, `fns-mmq-parity.exe`, `fns-attention-mma-parity.exe`,
  `fns-sample-check.exe`, `fns-mmvq-bench.exe`, `fns-exl3-parity.exe`: the test programs in [TESTING.md](TESTING.md)
- `fns-xgrammar-smoke.exe`: the native grammar smoke test

CMake copies the llama.cpp runtime DLLs next to every executable except `fns-xgrammar-smoke.exe`, which does not link
llama.cpp, so each one runs from `build\`.

## 4. Python environment

Install the Python dependencies into Python 3.14:

```powershell
py -3.14 -m pip install -r requirements-test.txt
```

- `requirements.txt` (included by `requirements-test.txt`): `tokenizers`, `jinja2`, `huggingface-hub`. Needed by the
  server, `serve/fns.py`, the Python tools and the CPU unit tests. Torch is not required.
- `requirements-test.txt` adds `numpy`, needed by `tests/kernel_parity.py` and `bench/scripts/route_profile.py`.
- `tests/kernel_parity.py` and `bench/scripts/route_profile.py hits` also import `gguf` from `gguf-py` in the pinned llama.cpp
  checkout (`FNS_LLAMA_DIR`). It is not installed by pip.
- xgrammar is not installed here. The optional reference environment is in [TESTING.md](TESTING.md#optional-xgrammar-reference).

Virtual environments inside the repository (`.venv\`) are git-ignored. To use one, create it with
`py -3.14 -m venv .venv` and run `.venv\Scripts\python -m pip install -r requirements-test.txt`.

## 5. Clean-tree verification

Before publishing, build a fresh clone that has the pinned checkouts from sections 1 and 2 and an empty build directory:

```powershell
$env:FNS_BUILD_DIR = "build-clean"
.\build.bat
```

Building needs no model file.

## Troubleshooting

- `find_package(CUDAToolkit)` fails: set `CUDA_PATH` to the CUDA 13.4 root.
- Configure stops with `llama.cpp Release build: missing file`: run `bench\scripts\build-llama.bat`, or set `FNS_LLAMA_BUILD` to the Release build of the pinned llama.cpp.
- Configure stops with `xgrammar: missing file`: run `tools\xgrammar\build.bat` after the pinned checkout and submodule update, or set `FNS_XGRAMMAR_BUILD`.
- `ninja.exe` not found: add Ninja to `PATH` or set `FNS_NINJA` to its full path.
- `build.bat` reports a missing MSVC toolchain: install the Visual Studio 2022 C++ workload, or run from an x64 Native Tools prompt.
