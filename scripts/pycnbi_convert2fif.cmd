@echo off
if not "%1" == "" (
    python -c "if __name__ == '__main__': from pycnbi.utils.convert2fif import main; main('%1')"
) else (
    echo Usage: %0 {fif_path}
)
