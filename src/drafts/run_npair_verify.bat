@echo off
rem n-pair verification, n = 1..6, against run_0910_overnight's step-7 result.
rem CSVs and PNGs land in OUT.  ~1.5 h on 7 pairs (2-8, ref 2).
rem A failed n is reported and the next one still runs.

setlocal EnableDelayedExpansion
set "ROOT=%~dp0..\.."
set "PY=C:\Users\weishan\.conda\envs\rbonn\python.exe"
set "SCRIPT=%ROOT%\src\calibration_module\steps\calib_npair_verify.py"
set "RUN=%ROOT%\src\calib_data\run_0910_overnight"
set "STEP7=%RUN%\calib_step7v2_0911_0031.json"
set "OUT=%RUN%"
set "FAILED="

for %%N in (1 2 3 4 5 6) do (
    echo.
    echo ==================== n = %%N  [!date! !time!] ====================
    "%PY%" "%SCRIPT%" --step7 "%STEP7%" --n %%N --out "%OUT%"
    if errorlevel 1 (
        echo *** n = %%N failed
        set "FAILED=!FAILED! %%N"
    )
)

echo.
echo ==================== done  [%date% %time%] ====================
if defined FAILED (echo failed n:%FAILED%) else (echo all n = 1..6 completed)
endlocal
