@echo off
setlocal enabledelayedexpansion

python %NEUROD_ROOT%\neurodecode\utils\io\fif2mat.py %1 %2
