# Assign NEUROD_SCRIPTS
clear
echo ========================================================================================
echo Please type the path to folder which will contain the subjects scripts and press ENTER
echo 
read -p 'Path: ' FOLDER

:choice
if test -d "$FOLDER"; then
    echo 'export NEUROD_SCRIPTS='$FOLDER >> ~/.profile 
else
    echo
    echo The provided path does not exit!
     goto :choice) 
fi

# Assign NEUROD_DATA
echo
echo ========================================================================================
echo Please type the path to folder which will contain the subjects data and press ENTER.
echo
read -p 'Path: ' FOLDER

:choice
if test -d "$FOLDER"; then
    echo 'export NEUROD_DATA='$FOLDER >> ~/.profile 
else
    echo
    echo The provided path does not exit!
    goto :choice
fi

# Assign NEUROD_ROOT
echo 'export NEUROD_ROOT='$PWD >> ~/.profile 

# Install neurodecode
echo
echo ========================================================================================
echo Installing dependencies
echo
version=$(python -V 2>&1 | grep -Po '(?<=Python )(.)')
@set /P "c=Do you want to install in dev mode [Y/N]?"
if [/I "%c%" EQU "Y"]; then
    dev="-e"
else if [/I "%c%" EQU "N"];
    dev=""
else
    dev=""
    echo Unknown answer, will install NeuroDecode normally.
fi
if [ "$version" -gt "2" ]; then
	pip3 install $dev .
else
	pip2 install $dev .
fi

# Add to PATH the scripts folder
echo 'alias neurodecode=$NEUROD_ROOT"/scripts/unix/nd_gui.sh"' >> ~/.bashrc
source ~/.profile
source ~/.bashrc
chmod u+x $NEUROD_ROOT"/scripts/unix/nd_gui.sh"