@echo off
rem Reference build of llama.cpp for benchmarks: pinned commit 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b (docs/BUILD.md).
rem CUDA sm_86 only, Release, flash attention for all KV quant pairs (F16 and q8_0 are measured).
rem Environment: FNS_LLAMA_DIR (checkout, default external\llama.cpp), FNS_LLAMA_BUILD (build dir, default FNS_LLAMA_DIR\build-sm86),
rem FNS_NINJA, CUDA_PATH. Extra arguments are passed to cmake as -D options.
call "%~dp0..\..\tools\build-env.bat" || exit /b 1
if not defined FNS_LLAMA_BUILD set "FNS_LLAMA_BUILD=%FNS_LLAMA_DIR%\build-sm86"
if not exist "%FNS_LLAMA_DIR%\CMakeLists.txt" (
  echo ERROR: llama.cpp checkout not found at "%FNS_LLAMA_DIR%". Clone it at commit 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b or set FNS_LLAMA_DIR.
  exit /b 1
)
where git >nul 2>&1 || (echo ERROR: git not found on PATH. & exit /b 1)
where cmake >nul 2>&1 || (echo ERROR: cmake not found on PATH. & exit /b 1)

cmake -G Ninja -S "%FNS_LLAMA_DIR%" -B "%FNS_LLAMA_BUILD%" ^
  -DCMAKE_MAKE_PROGRAM="%FNS_NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real ^
  -DCMAKE_CUDA_COMPILER="%FNS_NVCC%" ^
  -DCUDAToolkit_ROOT="%FNS_CUDA_ROOT%" ^
  -DGGML_NATIVE=ON "-DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0" ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF %* || exit /b 1
cmake --build "%FNS_LLAMA_BUILD%" --target llama-bench llama-server llama-perplexity llama-cli -j 16 || exit /b 1
git -C "%FNS_LLAMA_DIR%" rev-parse HEAD
