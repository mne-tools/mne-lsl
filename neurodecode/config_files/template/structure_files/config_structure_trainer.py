########################################################################
class Basic:
    """
    Add the basic parameters for your trainer modality. 

    These parameters will be displayed on the Basic page of the GUI. 

    This file defines the parameter type and-or its possible values displayed on the GUI, 
    it is not used outside the GUI. Only the config_template are loaded and used in the protocol.

    Each paramsX dict will display all the parameters in the same layout on the GUI page (a separator line between each part)

    tuple   :   all the possible to chose from
    str     :   a string input
    int     :   an integer input
    float   :   an floating input
    list    :   multiple non-fixed values in a list (int, float, str) 
    dict    :   for a parameter with internal params
    
    Parameter name contains _PATH  + str        :   Allow to search a folder
    Parameter name contains _FILE + str         :   Allow to search a file
    Parameter name contains _CHANNELS + list    :   Load the channels list from a file (channelsList.txt) 
                                                    for selected the channels of interest
                                                    
    Check the MI config_file for examples
    """  

    #-------------------------------------------
    # Layout 1
    #-------------------------------------------
    params0 = dict()
    params0.update({'XXX_PATH': str})
    
    #-------------------------------------------
    # Layout 2 
    #-------------------------------------------
    params1 = dict()
    params1.update({'AAA': (None, 'A', 'B', 'C', 'D', 'E')})
    params1.update({'BBB': (False, True)})
    params1.update({'CCC': int})
    params1.update({'DDD': float})
    params1.update({'EEE': str})
    
    #-------------------------------------------
    # Layout X
    #-------------------------------------------
    params2 = dict()

########################################################################
class Advanced:
    """
    Contains the advanced parameters for your offline modality.

    These parameters will be displayed on the Advanced page of the GUI.

    Same as above
    """

    #-------------------------------------------
    # # Layout 1
    #-------------------------------------------
    params1 = dict()
    params1.update({'XXX_FILE': str})

    #-------------------------------------------
    # Layout 2
    #-------------------------------------------
    params2 = dict()
    params2.update({'AAA':dict(A=float, B=int, C=str, D=list, E=tuple)})

    #-------------------------------------------
    # Layout X
    #-------------------------------------------
    params3 = dict()
