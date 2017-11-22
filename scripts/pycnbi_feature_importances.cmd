@ECHO OFF
python -c "if __name__ == '__main__': import pycnbi.utils.feature_importances as m; m.config_run('%1', '%2')"
