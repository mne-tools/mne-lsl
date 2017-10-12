@ECHO OFF
SET EXE="import pycnbi.decoder.trainer as m; m.config_run('%1')"
ipython -c %EXE%
