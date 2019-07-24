# ----------------------------------------------------------------------
def read_params_from_txt(folderPath, txtFile):
    """
    Loads the parameters from a txt file.
    
    path = folder path 
    txtFile = file containing the params to load
    """
    file = open(folderPath + '/' + txtFile)
    params = file.read().splitlines()
    file.close()
    
    return params

# ----------------------------------------------------------------------
def save_params_to_txt(outdir, txtFile, params):
    """
    Save the params to a txt file for the GUI
    
    outdir = folder path
    txtFile = txt file name
    params = params to save
    """
    filename = outdir + "channelsList.txt"
    config = Path(filename)
    
    if config.is_file() is False:
        file = open(filename, "w")    
        for x in range(len(params)):
            file.write(params[x] + "\n")
        file.close()
