@echo off
setlocal enabledelayedexpansion

python %NEUROD_ROOT%\neurodecode\utils\io\convert2fif.py %1 %2
