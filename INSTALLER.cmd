@REM Assign NEUROD_SCRIPTS
cls
@echo ========================================================================================
@echo Please type the path to folder which will contain the subjects' scripts and press ENTER.
@echo.
@set /P "FOLDER=Path: "
@IF exist %FOLDER% (
    @setx NEUROD_SCRIPTS %FOLDER%
) ELSE (
    @echo.
    @echo The provided path does not exit!
    @echo ==== Installation stopped ==== 
    @ping 127.0.0.1 -n 4 > nul
    @Exit
    ) 

@REM Assign NEUROD_DATA
@echo.
@echo ========================================================================================
@echo Please type the path to folder which will contain the subjects' data and press ENTER.
@echo.
@set /P "FOLDER=Path: "
@IF exist %FOLDER% (
    @setx NEUROD_DATA %FOLDER%
) ELSE (
    @echo.
    @echo The provided path does not exit!
    @echo ==== Installation stopped ====
    @ping 127.0.0.1 -n 4 > nul
    @Exit) 

@REM Assign NEUROD_ROOT
@setx NEUROD_ROOT %cd%

@REM Install neurodecode
@echo.
@echo ========================================================================================
@echo Installing dependencies
@echo.
pip install -e .

@REM Add to PATH the scripts folder
@set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive -NoProfile
%PWS% -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User')+';'+$env:NEUROD_ROOT+'\scripts', 'User')"

@REM Create shorcut on the Desktop
@set TARGET='%NEUROD_ROOT%\scripts\nd_gui.cmd'
@set SHORTCUT='%userprofile%\Desktop\neurodecode.lnk'
@set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive -NoProfile
%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(%SHORTCUT%); $S.TargetPath = %TARGET%; $S.Save()"