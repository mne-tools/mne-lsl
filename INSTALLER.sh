# Assign NEUROD_SCRIPTS
clear
echo ========================================================================================
echo Please type the path to folder which will contain the subjects scripts and press ENTER
echo 
read -p 'Path: ' FOLDER

if test -f "$FOLDER"; then
    echo 'export NEUROD_SCRIPTS=$FOLDER' >> ~/.bashrc 
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

if test -f "$FOLDER"; then
    echo 'export NEUROD_DATA=$FOLDER' >> ~/.bashrc 
else
    echo
    echo The provided path does not exit!
    echo ==== Installation stopped ==== 
    ping 127.0.0.1 -n 4 > nul
    exit
fi

# Assign NEUROD_ROOT
echo 'export NEUROD_ROOT=$PWD' >> ~/.bashrc 

# Install neurodecode
echo
echo ========================================================================================
echo Installing dependencies
echo
pip install -e .

# Add to PATH the scripts folder
echo 'export PATH="$PATH:$NEUROD_ROOT\scripts"' >> ~/.profile 

# Create shorcut on the Desktop
ln -ls $NEUROD_ROOT/scripts/unix/nd_gui.sh ~/Desktop/neurodecode