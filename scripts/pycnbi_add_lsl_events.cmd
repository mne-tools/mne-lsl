@ECHO OFF
IF NOT "%1" == "" (
    python -c "if __name__ == '__main__': import pycnbi.utils.add_lsl_events as m; m.add_lsl_events('%1')"
) ELSE (
    ECHO Usage: %0 [PATH_TO_EVENT_FILES]
)
