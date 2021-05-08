@echo off
setlocal enabledelayedexpansion

python %NEUROD_ROOT%/neurodecode/stream_recorder/stream_recorder.py %1 %2
