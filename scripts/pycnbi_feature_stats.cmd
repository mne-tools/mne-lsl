@ECHO OFF
python -c "if __name__ == '__main__': import pycnbi.analysis.parse_features as m; m.config_run('%1')"
