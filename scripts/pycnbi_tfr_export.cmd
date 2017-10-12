@ECHO OFF
SET EXE="import pycnbi.analysis.tfr_export as m; m.config_run('%1')"
ipython -c %EXE%
