import os
import sys
import shutil
import pickle
import numpy as np

from neurodecode import logger

#----------------------------------------------------------------------
def get_file_list(path, fullpath=True, recursive=False):
    """
    Get files with or without full path.
    
    Parameters
    ----------
    path : str
        The directory containing the files
    fullpath : bool
        If True, returns the file's absolute path
    recursive : bool
        If true, search recursively
        
    Returns
    -------
    list : The files' list
    """
    path = path.replace('\\', '/')
    if not path[-1] == '/': path += '/'

    if recursive == False:
        if fullpath == True:
            filelist = [path + f for f in os.listdir(path) if os.path.isfile(path + '/' + f) and f[0] != '.']
        else:
            filelist = [f for f in os.listdir(path) if os.path.isfile(path + '/' + f) and f[0] != '.']
    else:
        filelist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            if fullpath == True:
                [filelist.append(root + '/' + f) for f in files]
            else:
                [filelist.append(f) for f in files]
    return sorted(filelist)

#----------------------------------------------------------------------
def get_dir_list(path, recursive=False, no_child=False):
    """
    Get directory list relative to path.

    Parameters
    ----------
    path : str
        The directory to look at
    recusrive : bool
        If True, search recursively.
    no_child : bool
        If True, search directories having no child directory (leaf nodes)
        
    Returns
    -------
    list : The directories' list
    """
    path = path.replace('\\', '/')
    if not path[-1] == '/': path += '/'

    if recursive == True:
        pathlist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            [pathlist.append(root + '/' + d) for d in dirs]

            if no_child:
                for p in pathlist:
                    if len(get_dir_list(p)) > 0:
                        pathlist.remove(p)

    else:
        pathlist = [path + f for f in os.listdir(path) if os.path.isdir(path + '/' + f)]
        if no_child:
            for p in pathlist:
                if len(get_dir_list(p)) > 0:
                    pathlist.remove(p)

    return sorted(pathlist)

#----------------------------------------------------------------------
def make_dirs(dirname, delete=False):
    """
    Create a new directory
    
    Parameters
    ----------
    dirname : str
        The name of the new directory
    delete : bool
        If True, if the directory already exists, it will be deleted
    """
    if os.path.exists(dirname) and delete == True:
        try:
            shutil.rmtree(dirname)
        except OSError:
            logger.error('Directory was not completely removed. (Perhaps a Dropbox folder?). Continuing.')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#----------------------------------------------------------------------
def save_obj(fname, obj, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save the python object into a file
    
    Set protocol=2 for Python 2 compatibility
    
    Parameters
    ----------
    fname : str
        The file name where the python object is saved
    obj : python.Object
        The object to save
    protocol : int
        The pickle protocol
    """
    with open(fname, 'wb') as fout:
        pickle.dump(obj, fout, protocol)

#----------------------------------------------------------------------
def load_obj(fname):
    """
    Read a python object from a file
    
    Parameters
    ----------
    fname : str
        The file name 
    """
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        # usually happens when trying to load Python 2 pickle object from Python 3
        with open(fname, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except:
        msg = 'load_obj(): Cannot load pickled object file "%s". The error was:\n%s\n%s' %\
              (fname, sys.exc_info()[0], sys.exc_info()[1])
        raise IOError(msg)

#----------------------------------------------------------------------
def loadtxt_fast(filename, delimiter=',', skiprows=0, dtype=float):
    """
    Much faster matrix loading than numpy's loadtxt.
    
    Only work for simple and regular data (http://stackoverflow.com/a/8964779).
    Numpy's loadtxt do a lot of guessing and error-checking.
    
    Parameters
    ----------
    filename : str
        The txt file's path
    delimiter : str
        The data delimiter
    skiprows : int
        To skip the first rows
    dtype : python type
        Define the data type
        
    Returns
    -------
    numpy.Array
        The extracted data
    """
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        loadtxt_fast.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, loadtxt_fast.rowlength))
    
    return data

#----------------------------------------------------------------------
def parse_path(file_path):
    """
    Parse the file path with dir, name and extension
    
    Parameters
    ----------
    filepath : str
        The file's absolute path
    
    Returns
    -------
    python class : Its attributes are dir, name and ext.
    """
    class path_info:
        def __init__(self, path):
            path_abs = os.path.realpath(path).replace('\\', '/')
            s = path_abs.split('/')
            f = s[-1].split('.')
            basedir = '/'.join(s[:-1])
            if len(f) == 1:
                name, ext = f[-1], ''
            else:
                name, ext = '.'.join(f[:-1]), f[-1]
            self.dir = basedir
            self.name = name
            self.ext = ext
            self.txt = 'self.dir=%s\nself.name=%s\nself.ext=%s\n' % (self.dir, self.name, self.ext)
        def __repr__(self):
            return self.txt
        def __str__(self):
            return self.txt

    return path_info(file_path)

#----------------------------------------------------------------------
def forward_slashify(txt):
    """
    Replace all the backslash to slash for python compatibility
    
    Parameters
    ----------
    txt : str
        The path to slashify
    
    Returns
    -------
    str : The slashified path
    """
    return txt.replace('\\\\', '/').replace('\\', '/')
