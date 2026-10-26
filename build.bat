@echo off
rem Release build of FlintNSteel: MSVC 2022, CUDA Toolkit, Ninja, sm_86.
rem Usage: build.bat [-DNAME=VALUE ...]   (extra -D options override the defaults)
rem Environment: FNS_BUILD_DIR, FNS_LLAMA_DIR, FNS_LLAMA_BUILD, FNS_XGRAMMAR_DIR, FNS_XGRAMMAR_BUILD, FNS_NINJA, CUDA_PATH.
call "%~dp0tools\build-env.bat" || exit /b 1
where cmake >nul 2>&1 || (echo ERROR: cmake not found on PATH. & exit /b 1)

cmake -G Ninja -S "%FNS_ROOT%" -B "%FNS_BUILD_DIR%" ^
  -DCMAKE_MAKE_PROGRAM="%FNS_NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real ^
  -DCMAKE_CUDA_COMPILER="%FNS_NVCC%" ^
  -DCUDAToolkit_ROOT="%FNS_CUDA_ROOT%" ^
  -DFNS_LLAMA_DIR="%FNS_LLAMA_DIR%" -DFNS_XGRAMMAR_DIR="%FNS_XGRAMMAR_DIR%" %* || exit /b 1
cmake --build "%FNS_BUILD_DIR%" -j 16 || exit /b 1
