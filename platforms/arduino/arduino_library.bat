@echo off
setlocal

if not exist "src" mkdir "src"
if not exist "examples" mkdir "examples"

del /Q "src\*.c" 2>nul
del /Q "src\*.h" 2>nul

xcopy "..\..\src" "src\" /E /I /Y >nul
xcopy "..\..\examples" "examples\" /E /I /Y >nul

copy /Y "aifes_config.cpp" "src\" >nul
copy /Y "aifes_config.h" "src\" >nul

echo Done.
pause