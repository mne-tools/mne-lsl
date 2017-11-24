@ECHO OFF
echo '%1'
IF NOT "%1" == "" (
    python -c "if __name__ == '__main__': import pycnbi.decoder.trainer as m; m.config_run('%1')"
    pause
) ELSE (
    ECHO Usage: %0 [CONFIG_FILE]
)
