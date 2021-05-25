@echo off
setlocal enabledelayedexpansion

python %NEUROD_ROOT%\neurodecode\utils\io\preprocess\mne\resample.py %1 %2 %3
