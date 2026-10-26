# FlintNSteel: Flint inference engine for CUDA sm_86 on Windows (MSVC, Ninja).
# Dependency directories are CACHE PATH variables. Each defaults to its environment variable when set, otherwise to a
# path under external/ in this repository. A -D value on the command line always takes precedence.
#   FNS_LLAMA_DIR, FNS_LLAMA_BUILD      llama.cpp checkout and its Release build (bench/scripts/build-llama.bat)
#   FNS_XGRAMMAR_DIR, FNS_XGRAMMAR_BUILD xgrammar checkout and its build (tools/xgrammar/build.bat)
cmake_minimum_required(VERSION 3.28)
project(FlintNSteel LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86-real)
endif()

# Sets ${out} to the environment variable ${var} (as a CMake path) if it is set and non-empty, else to ${fallback}.
function(fns_path_default out var fallback)
    if(DEFINED ENV{${var}} AND NOT "$ENV{${var}}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{${var}}" _p)
    else()
        set(_p "${fallback}")
    endif()
    set(${out} "${_p}" PARENT_SCOPE)
endfunction()

# Fails configuration with a message naming the missing file and the step that produces it.
function(fns_require_files what hint)
    foreach(_f ${ARGN})
        if(NOT EXISTS "${_f}")
            message(FATAL_ERROR "${what}: missing file\n  ${_f}\n${hint}")
        endif()
    endforeach()
endfunction()

fns_path_default(_llama_dir FNS_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
set(FNS_LLAMA_DIR "${_llama_dir}" CACHE PATH "llama.cpp checkout at the pinned commit (docs/BUILD.md)")
fns_path_default(_llama_build FNS_LLAMA_BUILD "${FNS_LLAMA_DIR}/build-sm86")
set(FNS_LLAMA_BUILD "${_llama_build}" CACHE PATH "llama.cpp Release build directory of FNS_LLAMA_DIR")

fns_path_default(_xgrammar_dir FNS_XGRAMMAR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/xgrammar")
set(FNS_XGRAMMAR_DIR "${_xgrammar_dir}" CACHE PATH "xgrammar checkout at the pinned v0.2.1 commit (docs/BUILD.md)")
fns_path_default(_xgrammar_build FNS_XGRAMMAR_BUILD "${FNS_XGRAMMAR_DIR}/build-fns")
set(FNS_XGRAMMAR_BUILD "${_xgrammar_build}" CACHE PATH "xgrammar build directory containing xgrammar.lib")

set(_gg "${FNS_LLAMA_DIR}/ggml")

# llama.cpp: sources of the ggml-cuda kernels compiled here, and libraries/DLLs of the Release build.
set(_mmq_types q8_0 q4_k q5_k q6_k iq4_xs iq3_s iq3_xxs iq2_s)
set(_mmq_inst)
foreach(_t ${_mmq_types})
    list(APPEND _mmq_inst ${_gg}/src/ggml-cuda/template-instances/mmq-instance-${_t}.cu)
endforeach()
set(_fns_llama_dlls ggml-base ggml ggml-cpu ggml-cuda llama)

fns_require_files("llama.cpp checkout" "Clone llama.cpp at commit 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b and set FNS_LLAMA_DIR to it."
    "${FNS_LLAMA_DIR}/include/llama.h"
    "${_gg}/include/ggml.h"
    "${_gg}/src/ggml-cuda/quantize.cu"
    "${_gg}/src/ggml-cuda/mmid.cu"
    "${_gg}/src/ggml-cuda/convert.cu"
    ${_mmq_inst})

set(_fns_llama_build_files
    "${FNS_LLAMA_BUILD}/ggml/src/ggml-base.lib"
    "${FNS_LLAMA_BUILD}/ggml/src/ggml.lib"
    "${FNS_LLAMA_BUILD}/src/llama.lib")
foreach(_dll ${_fns_llama_dlls})
    list(APPEND _fns_llama_build_files "${FNS_LLAMA_BUILD}/bin/${_dll}.dll")
endforeach()
fns_require_files("llama.cpp Release build" "Run bench/scripts/build-llama.bat (set FNS_LLAMA_DIR first if the checkout is elsewhere), or set FNS_LLAMA_BUILD to the build directory."
    ${_fns_llama_build_files})

# xgrammar: headers and the static library are required; the grammar-constrained server is always built.
fns_require_files("xgrammar" "Clone xgrammar v0.2.1 (commit 5b4e9ce9e72524037ae24ecd831b9b6604d2eb48, with submodules) into FNS_XGRAMMAR_DIR and run tools/xgrammar/build.bat (or set FNS_XGRAMMAR_BUILD)."
    "${FNS_XGRAMMAR_DIR}/include/xgrammar/xgrammar.h"
    "${FNS_XGRAMMAR_DIR}/3rdparty/picojson/picojson.h"
    "${FNS_XGRAMMAR_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h"
    "${FNS_XGRAMMAR_BUILD}/xgrammar.lib")

find_package(CUDAToolkit REQUIRED)

add_library(fns_xgrammar INTERFACE)
target_include_directories(fns_xgrammar INTERFACE
    "${FNS_XGRAMMAR_DIR}/include" "${FNS_XGRAMMAR_DIR}/3rdparty/dlpack/include" "${FNS_XGRAMMAR_DIR}/3rdparty/picojson")
target_link_libraries(fns_xgrammar INTERFACE "${FNS_XGRAMMAR_BUILD}/xgrammar.lib")

add_library(fns_ggml_base SHARED IMPORTED)
set_target_properties(fns_ggml_base PROPERTIES
    IMPORTED_IMPLIB "${FNS_LLAMA_BUILD}/ggml/src/ggml-base.lib"
    IMPORTED_LOCATION "${FNS_LLAMA_BUILD}/bin/ggml-base.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${_gg}/include"
    INTERFACE_COMPILE_DEFINITIONS GGML_SHARED)

set(FNS_CUDA_FLAGS -use_fast_math -extended-lambda -Xcompiler=/Zc:preprocessor -Xcompiler=/bigobj
    -Xcompiler=/utf-8 -diag-suppress=177,550)

# ggml-cuda kernels called directly (MMVQ, MMQ, Q8_1 quantization, MUL_MAT_ID sorting) and minimal host symbols.
add_library(fns_ggml STATIC src/ggml/ggml_mmvq.cu src/ggml/ggml_mmq.cu src/ggml/ggml_fattn.cu src/ggml/ggml_cuda_host.cu
    "${_gg}/src/ggml-cuda/quantize.cu" "${_gg}/src/ggml-cuda/mmid.cu" "${_gg}/src/ggml-cuda/convert.cu" ${_mmq_inst})
target_include_directories(fns_ggml PRIVATE "${_gg}/src/ggml-cuda" "${_gg}/src" PUBLIC include)
target_compile_options(fns_ggml PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FNS_CUDA_FLAGS}>)
target_link_libraries(fns_ggml PUBLIC fns_ggml_base CUDA::cudart)

add_library(fns STATIC src/model.cpp src/engine.cu src/kernels.cu src/prefill_kernels.cu src/fa_int8.cu src/attention_mma.cu)
target_link_libraries(fns PUBLIC CUDA::cublas)
target_include_directories(fns PUBLIC include)
target_compile_options(fns PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -Xcompiler=/utf-8>
                                   $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
target_link_libraries(fns PUBLIC fns_ggml)

add_executable(fns-ref tools/fns_ref/fns_ref.cpp)
target_compile_options(fns-ref PRIVATE /utf-8)
target_link_libraries(fns-ref PRIVATE fns)

add_executable(fns-bench tools/fns_bench/fns_bench.cpp)
target_compile_options(fns-bench PRIVATE /utf-8)
target_link_libraries(fns-bench PRIVATE fns)

# Quality against the Q8_0 baseline from llama-perplexity (same KL / top-1 / PPL protocol).
add_executable(fns-kld tools/fns_kld/fns_kld.cpp)
target_compile_options(fns-kld PRIVATE /utf-8)
target_link_libraries(fns-kld PRIVATE fns)

# Multi-column verification (W columns): parity with sequential decoding, causality, greedy speculative decoding.
add_executable(fns-verify tools/fns_verify/fns_verify.cpp)
target_compile_options(fns-verify PRIVATE /utf-8)
target_link_libraries(fns-verify PRIVATE fns)

# Engine as a process driven by a server. Grammar constraints use the xgrammar C++ library (tools/xgrammar/build.bat).
# picojson (header-only, BSD-2) comes from the xgrammar checkout; the JSON request carries the nested grammar.
add_executable(fns-serve tools/fns_serve/fns_serve.cpp)
target_compile_options(fns-serve PRIVATE /utf-8)
target_compile_definitions(fns-serve PRIVATE FNS_WITH_XGRAMMAR)
target_link_libraries(fns-serve PRIVATE fns fns_xgrammar)

# Native grammar smoke test: xgrammar matcher on the Flint vocabulary (see tests/xgrammar_smoke.cpp for arguments).
add_executable(fns-xgrammar-smoke tests/xgrammar_smoke.cpp)
target_compile_options(fns-xgrammar-smoke PRIVATE /utf-8)
target_link_libraries(fns-xgrammar-smoke PRIVATE fns_xgrammar)

# Sampling: draw frequencies against the top-k / top-p / temperature distribution.
add_executable(fns-sample-check tests/sample_check.cu)
target_compile_options(fns-sample-check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-sample-check PRIVATE fns)

add_executable(fns-mmvq-bench tests/mmvq_bench.cu)
target_compile_options(fns-mmvq-bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-mmvq-bench PRIVATE fns)

# Quantized GEMV against gguf-py dequantization (tests/kernel_parity.py).
add_executable(fns-kernel-parity tests/kernel_parity.cu)
target_compile_options(fns-kernel-parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-kernel-parity PRIVATE fns)

# Expert pointer tables (VRAM and zero-copy) against ggml mmvq_ids, bit for bit.
add_executable(fns-expert-parity tests/expert_parity.cu)
target_compile_options(fns-expert-parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-expert-parity PRIVATE fns)

# Prefill MMQ: expert pointer tables against contiguous experts (bit for bit), and deviation from MMVQ (dense and experts).
add_executable(fns-mmq-parity tests/mmq_parity.cu)
target_compile_options(fns-mmq-parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-mmq-parity PRIVATE fns)

# Q/K/V from GGUF blocks: MMA attention against ggml flash attention and the sequential reference.
add_executable(fns-attention-mma-parity tests/attention_mma_parity.cu)
target_compile_options(fns-attention-mma-parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-attention-mma-parity PRIVATE fns)

# llama.cpp reference logits for the model parity gate (docs/TESTING.md).
add_executable(fns-dump tools/fns_dump/fns_dump.cpp)
target_compile_options(fns-dump PRIVATE /utf-8)
target_include_directories(fns-dump PRIVATE "${FNS_LLAMA_DIR}/include" "${_gg}/include")
target_link_libraries(fns-dump PRIVATE "${FNS_LLAMA_BUILD}/src/llama.lib" "${FNS_LLAMA_BUILD}/ggml/src/ggml.lib"
                                       "${FNS_LLAMA_BUILD}/ggml/src/ggml-base.lib")

# EXPERIMENTAL EXL3 path, independent of fns/GGUF compile options. Model parity is not validated; not recommended for serving.
add_library(fns_exl3 STATIC src/exl3/exl3_kernels.cu src/exl3/exl3_model.cpp)
target_include_directories(fns_exl3 PUBLIC src PRIVATE "${FNS_XGRAMMAR_DIR}/3rdparty/picojson")
target_link_libraries(fns_exl3 PUBLIC fns)
target_compile_options(fns_exl3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -Xcompiler=/utf-8 -Xcompiler=/bigobj -Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
add_executable(fns-exl3-parity tests/exl3_parity.cu)
target_include_directories(fns-exl3-parity PRIVATE "${FNS_XGRAMMAR_DIR}/3rdparty/picojson")
target_compile_options(fns-exl3-parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
target_link_libraries(fns-exl3-parity PRIVATE fns_exl3)
target_sources(fns_exl3 PRIVATE src/exl3/exl3_engine.cu)
target_link_libraries(fns-bench PRIVATE fns_exl3)
target_link_libraries(fns-serve PRIVATE fns_exl3)
target_link_libraries(fns-kld PRIVATE fns_exl3)

# llama.cpp runtime DLLs are copied next to every executable that links the engine, so each one runs from the build directory.
function(fns_copy_llama_runtime target)
    foreach(_dll ${_fns_llama_dlls})
        add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FNS_LLAMA_BUILD}/bin/${_dll}.dll" "$<TARGET_FILE_DIR:${target}>")
    endforeach()
endfunction()

foreach(_exe fns-ref fns-bench fns-kld fns-verify fns-serve fns-sample-check fns-mmvq-bench fns-kernel-parity
             fns-expert-parity fns-mmq-parity fns-attention-mma-parity fns-dump fns-exl3-parity)
    fns_copy_llama_runtime(${_exe})
endforeach()
