@echo off
if not "%1" == "" (
    python -c "if __name__ == '__main__': import pycnbi.utils.add_lsl_events as m; m.add_lsl_events('%1')"
    pause
) else (
    echo Usage: %0 {path_to_event_files}
)
