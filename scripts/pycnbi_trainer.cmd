@echo off
if not "%1" == "" (
    python -c "if __name__ == '__main__': import pycnbi.decoder.trainer as m; m.batch_run(r'%1')"
    pause
) else (
    echo Usage: %0 {config_file}
)
