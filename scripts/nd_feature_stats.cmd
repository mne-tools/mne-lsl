@echo off
python -c "if __name__ == '__main__': import neurodecode.analysis.parse_features as m; m.config_run(r'%1')"
