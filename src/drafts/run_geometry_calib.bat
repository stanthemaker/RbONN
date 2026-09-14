@echo off
rem For each geometry folder in RUNS, one after another:
rem   step 6 v2  pairs 1-8
rem   step 7 v2  targets 3-8 against reference 2
rem   n-pair verify, n = 1..7, over every pair step 7 calibrated (2-8)
rem starting once the running n-pair verify (run_npair_verify.bat) has let go
rem of the SLM and DAQ.
rem
rem Each folder must already hold exactly ONE calib_step3c_*.json: the 0910
rem step 3 with channel_width_px / gap_px edited (width + gap = 45).  Every
rem output lands in that folder.
rem ~2.5 h per geometry: step 6 ~44 min, step 7 ~11 min, verify ~1.5 h.
rem A failed step 6 or 7 skips the rest of that geometry; a failed n does not.

setlocal EnableDelayedExpansion
set "ROOT=%~dp0..\.."
set "PY=C:\Users\weishan\.conda\envs\rbonn\python.exe"
set "STEPS=%ROOT%\src\calibration_module\steps"
set "DATA=%ROOT%\src\calib_data"
set "RUNS=run_0911_10pairs_w15g35"
set "PAIRS6=1,2,3,4,5,6,7,8,9,10"
set "TARGETS7=2,3,4,5,6,7,8,9,10"
set "REF=1"
set "VERIFY_N=10"
rem wait while any cmd/python process has this in its command line
set "WAIT_FOR=npair_verify"
set "PYTHONIOENCODING=utf-8"
set "FAILED="

rem ---- check every folder now, so a missing copy shows before the wait
set "BAD="
for %%R in (%RUNS%) do (
    set "N=0"
    for %%F in ("%DATA%\%%R\calib_step3c_*.json") do set /a N+=1
    if not "!N!"=="1" (
        echo *** %DATA%\%%R needs exactly one calib_step3c_*.json, found !N!
        set "BAD=1"
    )
)
if defined BAD exit /b 1

echo Waiting for *%WAIT_FOR%* to finish  [%date% %time%]
powershell -NoProfile -Command "while (Get-CimInstance Win32_Process | Where-Object { ($_.Name -in 'cmd.exe','python.exe') -and ($_.CommandLine -like '*%WAIT_FOR%*') }) { Start-Sleep -Seconds 30 }"

for %%R in (%RUNS%) do call :geometry %%R

echo.
echo ==================== done  [%date% %time%] ====================
if defined FAILED (echo failed:%FAILED%) else (echo all geometries completed)
endlocal
exit /b 0


rem ======================================================================
rem :geometry NAME -- step 6 -> step 7 -> verify into %DATA%\NAME
:geometry
set "NAME=%~1"
set "DIR=%DATA%\%~1"
for %%F in ("%DIR%\calib_step3c_*.json") do set "STEP3=%%~fF"

echo.
echo ==================== %NAME%: step 6, pairs %PAIRS6%  [%date% %time%] ====================
call :newest "%DIR%" "calib_step6v2_result_*.json" BEFORE
"%PY%" "%STEPS%\calib_step6_v2.py" --step3 "%STEP3%" --pairs "%PAIRS6%" --out "%DIR%"
set "RC=%errorlevel%"
call :newest "%DIR%" "calib_step6v2_result_*.json" STEP6
if not "%RC%"=="0" (set "WHY=step 6 exited with %RC%" & goto :geometry_failed)
if "%STEP6%"=="%BEFORE%" (set "WHY=step 6 wrote no result JSON" & goto :geometry_failed)

echo.
echo ==================== %NAME%: step 7, targets %TARGETS7% ref %REF%  [%date% %time%] ====================
call :newest "%DIR%" "calib_step7_result_*.json" BEFORE
"%PY%" "%STEPS%\calib_step7_v2.py" --step6 "%STEP6%" --targets "%TARGETS7%" --ref %REF% --out "%DIR%"
set "RC=%errorlevel%"
call :newest "%DIR%" "calib_step7_result_*.json" STEP7
if not "%RC%"=="0" (set "WHY=step 7 exited with %RC%" & goto :geometry_failed)
if "%STEP7%"=="%BEFORE%" (set "WHY=step 7 wrote no result JSON" & goto :geometry_failed)

for %%N in (%VERIFY_N%) do (
    echo.
    echo ==================== %NAME%: verify n = %%N  [!date! !time!] ====================
    "%PY%" "%STEPS%\calib_npair_verify.py" --step7 "%STEP7%" --n %%N --out "%DIR%"
    if errorlevel 1 (
        echo *** %NAME%: verify n = %%N failed
        set "FAILED=!FAILED! %NAME%:n%%N"
    )
)
exit /b 0

:geometry_failed
echo *** %NAME%: %WHY% -- skipping the rest of this geometry
set "FAILED=%FAILED% %NAME%"
exit /b 1


rem :newest DIR PATTERN VAR -- VAR = full path of the newest match, or empty
:newest
set "%~3="
for /f "delims=" %%F in ('dir /b /a-d /o-d "%~1\%~2" 2^>nul') do if not defined %~3 set "%~3=%~1\%%F"
exit /b 0
