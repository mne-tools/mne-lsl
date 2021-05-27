@echo off
setlocal enabledelayedexpansion

python %NEUROD_ROOT%/scripts/python/nd_stream_recorder.py %1 %2
