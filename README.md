# FlintNSteel

FlintNSteel is a CUDA inference engine for one model, [Recursive Flint](https://huggingface.co/RecursiveSI/flint),
on one target: NVIDIA sm_86 (RTX 3090 class), Windows, batch size 1 (one agent session). It runs the GGUF
checkpoints from Unsloth, with an OpenAI-compatible server (`serve/fns_server.py`) for agent harnesses.

It is not a general runtime. Other models, other GPU architectures, and other platforms are not supported.

## Status

- GGUF path (`UD-Q3_K_XL` fully in VRAM, `UD-Q4_K_XL` with a hybrid expert cache): supported.
- `/v1/chat/completions` (streamed and non-streamed), tool-call grammar through xgrammar, and
  `reasoning_content`: supported.
- Speculative verification: supported; it produces the same tokens as sequential decoding. `serve/fns_server.py` enables it by default (`--spec 3`). The native `fns-serve` binary has it off unless `--spec` is given.
- Options that change output bits (`--kv-global q8_0`, `--fa-int8`, `--fa-swa`, `--attention-mma`): opt-in only,
  off by default. See [docs/ENGINE.md](docs/ENGINE.md).
- EXL3 weights (`--weights exl3`): **experimental and unvalidated.** Isolated kernels match their reference, but the
  full-model forward pass does not yet match ExLlamaV3. It is not recommended for serving.

## Documentation

| Document | Contents |
|---|---|
| [docs/BUILD.md](docs/BUILD.md) | Prerequisites, dependency pins, environment variables, build steps |
| [docs/USAGE.md](docs/USAGE.md) | Models, tokenizer, the server, the launcher, native tools |
| [docs/ENGINE.md](docs/ENGINE.md) | Engine design, options, opt-in incompatibilities, limits |
| [docs/TESTING.md](docs/TESTING.md) | CPU unit tests, the HTTP smoke test, GPU test programs, the Q4 routing profile and corpus, CI scope |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Third-party code, provenance and licenses |

## Quick start

Requires Windows 10/11, an sm_86 GPU, CUDA Toolkit 13.4, Visual Studio 2022 (MSVC), CMake, Ninja, Git and Python 3.14.
The pinned llama.cpp and xgrammar checkouts must be built first; see [docs/BUILD.md](docs/BUILD.md).

```powershell
bench\scripts\build-llama.bat                                # after cloning llama.cpp (pinned commit)
tools\xgrammar\build.bat                                     # after cloning xgrammar (pinned commit, submodules)
.\build.bat                                                  # FlintNSteel, output in build\
py -3.14 -m pip install -r requirements-test.txt
py -3.14 bench\scripts\fetch-models.py q3                    # models\flint-UD-Q3_K_XL.gguf
py -3.14 tools\xgrammar\export_vocab.py                      # models\prof\ (tokenizer download on first use)
py -3.14 serve\fns_server.py --model models\flint-UD-Q3_K_XL.gguf --port 8080
```

Then point an OpenAI-compatible client at `http://127.0.0.1:8080/v1`. See [docs/USAGE.md](docs/USAGE.md) for the
server options and the launcher (`serve\fns.py`), and [docs/TESTING.md](docs/TESTING.md) for the tests.

## License

Apache License 2.0. See [LICENSE](LICENSE). Third-party components keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
