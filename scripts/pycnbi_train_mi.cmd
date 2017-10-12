@ECHO OFF
SET EXE="import pycnbi.protocols.train_mi as m; m.config_run('%1')"
ipython -c %EXE%
