@echo off
setlocal enabledelayedexpansion

python %NEUROD_ROOT%\neurodecode\utils\io\fif_resample.py %1 %2 %3