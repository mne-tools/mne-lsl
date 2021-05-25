import os
import shutil
from pathlib import Path
from neurodecode import logger


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
    list : The file's list.
    """
    path = Path(path)
    if not path.exists():
        raise IOError(f"The directory '{path}' does not exist.")

    if recursive == False:
        if fullpath == True:
            filelist = [str(path / file.name) for file in os.scandir(path)
                        if (path / file.name).is_file() and file.name[0] != '.']
        else:
            filelist = [file.name for file in os.scandir(path)
                        if (path / file.name).is_file() and file.name[0] != '.']

    else:
        filelist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            if fullpath == True:
                for file in files:
                    if file[0] != '.':
                        filelist.append(str(Path(root) / file))
            else:
                for file in files:
                    if file[0] != '.':
                        filelist.append(file)

    return sorted(filelist)


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
    path = Path(path)
    if not path.exists():
        raise IOError(f"The directory '{path}' does not exist.")

    if recursive == False:
        dirlist = [str(path / file.name) for file in os.scandir(path)
                   if (path / file.name).is_dir()]
        if no_child:
            dirlist2remove = [d for d in dirlist if len(get_dir_list(d)) > 0]
            dirlist = [d for d in dirlist if d not in dirlist2remove]

    else:
        dirlist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            for d in dirs:
                dirlist.append(str(Path(root) / d))

            if no_child:
                dirlist2remove = [
                    d for d in dirlist if len(get_dir_list(d)) > 0]
                dirlist = [d for d in dirlist if d not in dirlist2remove]

    return sorted(dirlist)


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
    dirname = Path(dirname)
    if dirname.exists() and delete == True:
        try:
            shutil.rmtree(dirname)
        except OSError:
            logger.error(
                'Directory was not completely removed. (Perhaps a Dropbox folder?). Continuing.')
    if not dirname.exists():
        os.makedirs(dirname)
