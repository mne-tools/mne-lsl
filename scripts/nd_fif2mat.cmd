@echo off
if not "%1" == "" (
    python -c "if __name__ == '__main__': from pycnbi.utils.fif2mat import fif2mat; fif2mat(r'%1')"
) else (
    echo Usage: %0 {fif_path}
)
