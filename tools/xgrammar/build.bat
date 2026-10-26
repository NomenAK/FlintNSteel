@echo off
rem Builds the xgrammar v0.2.1 C++ library for FlintNSteel's grammar-constrained fns-serve (MSVC 2022, Ninja, no Python or TVM-FFI bindings).
rem Pinned commit: 5b4e9ce9e72524037ae24ecd831b9b6604d2eb48 (v0.2.1), checkout with submodules, in FNS_XGRAMMAR_DIR.
rem Output: FNS_XGRAMMAR_BUILD\xgrammar.lib (default FNS_XGRAMMAR_DIR\build-fns). The top-level CMake build links it.
rem The checkout is only read; nothing is written inside it except the build directory.
rem The repository's config.cmake forces XGRAMMAR_BUILD_PYTHON_BINDINGS=ON; the config.cmake written to the build directory is read first.
call "%~dp0..\build-env.bat" || exit /b 1
set "XG_PIN=5b4e9ce9e72524037ae24ecd831b9b6604d2eb48"

if not exist "%FNS_XGRAMMAR_DIR%\include\xgrammar\xgrammar.h" (
  echo ERROR: xgrammar checkout not found at "%FNS_XGRAMMAR_DIR%". Clone v0.2.1 with submodules and set FNS_XGRAMMAR_DIR.
  exit /b 1
)
where git >nul 2>&1 || (echo ERROR: git not found on PATH. & exit /b 1)
where cmake >nul 2>&1 || (echo ERROR: cmake not found on PATH. & exit /b 1)

rem Warn when the checkout is not the pinned commit; the build still runs so a deliberate checkout can be tested.
for /f "delims=" %%H in ('git -C "%FNS_XGRAMMAR_DIR%" rev-parse HEAD 2^>nul') do set "XG_HEAD=%%H"
if not "%XG_HEAD%"=="%XG_PIN%" echo WARNING: xgrammar HEAD is %XG_HEAD%, expected pin %XG_PIN% (v0.2.1).

if not exist "%FNS_XGRAMMAR_BUILD%" mkdir "%FNS_XGRAMMAR_BUILD%" || exit /b 1
(
  echo set^(CMAKE_BUILD_TYPE Release^)
  echo set^(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF^)
  echo set^(XGRAMMAR_BUILD_CXX_TESTS OFF^)
  echo set^(XGRAMMAR_ENABLE_COVERAGE OFF^)
  echo set^(XGRAMMAR_ENABLE_CPPTRACE OFF^)
  echo set^(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF^)
) > "%FNS_XGRAMMAR_BUILD%\config.cmake" || exit /b 1

cmake -G Ninja -S "%FNS_XGRAMMAR_DIR%" -B "%FNS_XGRAMMAR_BUILD%" ^
  -DCMAKE_MAKE_PROGRAM="%FNS_NINJA%" || exit /b 1
cmake --build "%FNS_XGRAMMAR_BUILD%" -j 16 || exit /b 1
if not exist "%FNS_XGRAMMAR_BUILD%\xgrammar.lib" (
  echo ERROR: build finished without "%FNS_XGRAMMAR_BUILD%\xgrammar.lib". Check the xgrammar target name in its CMakeLists.txt.
  exit /b 1
)
echo xgrammar library: "%FNS_XGRAMMAR_BUILD%\xgrammar.lib"
echo Next: build the top-level project; target fns-xgrammar-smoke is built with it.
