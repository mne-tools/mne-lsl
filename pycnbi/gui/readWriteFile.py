import os
from pathlib import Path

# ----------------------------------------------------------------------
def read_params_from_file(folderPath, txtFile):
    """
    Loads the parameters from a txt file.
    
    path = folder path 
    txtFile = file containing the params to load
    """
    file = open(os.fspath(folderPath / txtFile))
    params = file.read().splitlines()
    file.close()
    
    return params

# ----------------------------------------------------------------------
def save_params_to_file(filePath, params):
    """
    Save the params to a txt file for the GUI
    
    txtPath = File where to save the params
    params = params to save
    """
    
    config = Path(filePath)
    params_name = params.__dict__
    
    if config.is_file() is True:
        os.remove(config)
        
    with open(config, "w") as file:
        for key, value in params_name.items():      
            if 'PATH' in key or 'FILE' in key:
                file.write("{} = r'{}'\n" .format(key, value))
            elif type(value) is str:
                file.write("{} = '{}'\n" .format(key, value))
            else:
                file.write("{} = {}\n" .format(key, value))