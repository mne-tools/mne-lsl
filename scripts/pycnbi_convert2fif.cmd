@echo off
if not "%1" == "" (
    python -c "if __name__ == '__main__': from pycnbi.utils.convert2fif import main; main(r'%1')"
) else (
    echo Usage: %0 {data path}
)
