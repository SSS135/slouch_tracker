@echo off
rem Launch Slouch Tracker (tauri dev) with WebView2 remote debugging on port 9222,
rem so scripts/native-cdp.mjs can screenshot / read console / click the real window.
setlocal
cd /d %~dp0..

rem Resolve vcvars64.bat portably: an explicit VCVARS64 override wins; otherwise
rem ask vswhere for the latest x64 MSVC toolset install; finally fall back to the
rem well-known Community install path.
set "VCVARS=%VCVARS64%"
if not defined VCVARS call :find_vcvars
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

call "%VCVARS%"
set WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222
echo === tauri dev + WebView2 remote debugging on http://127.0.0.1:9222 ===
call npm run tauri:dev
goto :eof

:find_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :eof
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL goto :eof
set "VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
goto :eof
