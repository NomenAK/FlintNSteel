@echo off
rem Shared environment for FlintNSteel build scripts.
rem Usage: call "%~dp0build-env.bat" || exit /b 1
rem Sets FNS_ROOT and the dependency and build directories (environment override, else the repository's external\ and build\),
rem activates the MSVC x64 toolchain (active developer shell, else vswhere), and resolves ninja.exe and nvcc.exe.
rem Overrides: FNS_BUILD_DIR, FNS_LLAMA_DIR, FNS_LLAMA_BUILD, FNS_XGRAMMAR_DIR, FNS_XGRAMMAR_BUILD,
rem            FNS_NINJA (full path to ninja.exe), CUDA_PATH (toolkit root), VCINSTALLDIR (developer shell).

for %%I in ("%~dp0..") do set "FNS_ROOT=%%~fI"
if not defined FNS_BUILD_DIR set "FNS_BUILD_DIR=%FNS_ROOT%\build"
if not defined FNS_LLAMA_DIR set "FNS_LLAMA_DIR=%FNS_ROOT%\external\llama.cpp"
if not defined FNS_LLAMA_BUILD set "FNS_LLAMA_BUILD=%FNS_LLAMA_DIR%\build-sm86"
if not defined FNS_XGRAMMAR_DIR set "FNS_XGRAMMAR_DIR=%FNS_ROOT%\external\xgrammar"
if not defined FNS_XGRAMMAR_BUILD set "FNS_XGRAMMAR_BUILD=%FNS_XGRAMMAR_DIR%\build-fns"

rem MSVC: reuse an active developer shell, otherwise locate Visual Studio with vswhere.
if defined VCINSTALLDIR goto msvc_ok
where cl >nul 2>&1 && goto msvc_ok
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto no_msvc
set "VSINSTALL="
for /f "usebackq delims=" %%P in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%P"
if not defined VSINSTALL goto no_msvc
if not exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" goto no_msvc
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || goto no_msvc
:msvc_ok

rem Ninja: FNS_NINJA if set, otherwise ninja.exe from PATH.
if defined FNS_NINJA goto ninja_check
for %%N in (ninja.exe) do set "FNS_NINJA=%%~$PATH:N"
:ninja_check
if not defined FNS_NINJA goto no_ninja
if not exist "%FNS_NINJA%" goto no_ninja

rem CUDA: nvcc from CUDA_PATH\bin, otherwise from PATH. FNS_CUDA_ROOT is the toolkit root.
set "FNS_NVCC="
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "FNS_NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined FNS_NVCC for %%N in (nvcc.exe) do set "FNS_NVCC=%%~$PATH:N"
if not defined FNS_NVCC goto no_cuda
for %%D in ("%FNS_NVCC%") do set "FNS_CUDA_ROOT=%%~dpD.."
for %%D in ("%FNS_CUDA_ROOT%") do set "FNS_CUDA_ROOT=%%~fD"
exit /b 0

:no_msvc
echo ERROR: MSVC x64 toolchain not found. Run from a "x64 Native Tools Command Prompt for VS 2022", or install Visual Studio Build Tools with the "Desktop development with C++" workload.
exit /b 1

:no_ninja
echo ERROR: ninja.exe not found. Add Ninja to PATH or set FNS_NINJA to its full path.
exit /b 1

:no_cuda
echo ERROR: nvcc.exe not found. Install the CUDA Toolkit and set CUDA_PATH, or add its bin directory to PATH.
exit /b 1
