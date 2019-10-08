@echo off
setlocal enabledelayedexpansion

if not "%1" == "" (
	set fif=%1
	set fif=!fif:\=/!
	if "%2" == "" (
		set chunk=16
	) else (
		set chunk=%2
	)
    python -c "if __name__ == '__main__': import pycnbi.stream_player.stream_player as m; m.stream_player('StreamPlayer', '!fif!', !chunk!)"
    pause
) else (
    echo Usage: %0 {fif_file} [chunk_size=16]
)
