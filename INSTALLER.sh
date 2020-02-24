# Assign NEUROD_SCRIPTS
clear
echo ========================================================================================
echo Please type the path to folder which will contain the subjects scripts and press ENTER
echo 
read -p 'Path: ' FOLDER

if test -d "$FOLDER"; then
    echo 'export NEUROD_SCRIPTS='$FOLDER >> ~/.bashrc 
else
    echo
    echo The provided path does not exit!
    echo ==== Installation stopped ==== 
    ping 127.0.0.1 -n 4 > nul
    exit
fi

# Assign NEUROD_DATA
echo
echo ========================================================================================
echo Please type the path to folder which will contain the subjects data and press ENTER.
echo
read -p 'Path: ' FOLDER

if test -d "$FOLDER"; then
    echo 'export NEUROD_DATA='$FOLDER >> ~/.bashrc 
else
    echo
    echo The provided path does not exit!
    echo ==== Installation stopped ==== 
    ping 127.0.0.1 -n 4 > nul
    exit
fi

# Assign NEUROD_ROOT
echo 'export NEUROD_ROOT='$PWD >> ~/.bashrc 

# Install neurodecode
echo
echo ========================================================================================
echo Installing dependencies
echo

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.)')
if [ "$version" -gt "2" ]; then
	pip3 install -e .
else
	pip2 install -e .
fi

# Add to PATH the scripts folder
chmod u+x $NEUROD_ROOT"/scripts/unix/nd_gui.sh"
echo 'alias neurodecode=$NEUROD_ROOT"/scripts/unix/nd_gui.sh"' >> ~/.bashrc
source ~/.bashrc