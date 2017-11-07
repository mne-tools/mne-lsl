@ECHO OFF
IF NOT "%1" == "" (
    python -c "if __name__ == '__main__': import pycnbi.analysis.tfr_export as m; m.config_run('%1')"
    pause
) ELSE (
    ECHO Usage: %0 [CONFIG_FILE]
)
