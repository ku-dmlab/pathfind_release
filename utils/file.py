import os
from typing import List, Union

from natsort import natsorted

FileTree = List[Union[str, "FileTree"]]


def get_files(parent_dir: str) -> list[str]:
    """Return a list of file paths.

    :param parent_dir: The directory to be scanned for files.
    :returns: A list of tuples each containing a file path.
    """

    entries = os.scandir(parent_dir)
    return [entry.path for entry in entries if entry.is_file()]


def get_dirs(parent_dir: str) -> list[str]:
    """Return a list of subdirectory paths.

    :param parent_dir: The directory to be scanned for subdirectories.
    :returns: A list of tuples each containing a subdirectory path.
    """

    entries = os.scandir(parent_dir)
    return [entry.path for entry in entries if entry.is_dir()]


def tree(parent_dir: str) -> FileTree:
    """Return a nested list of files and directories recursively.

    :param parent_dir: The directory to be scanned.
    :param returns: A nested list of files and directories.
    """

    sorted = natsorted(p.path for p in os.scandir(parent_dir))
    return [p if os.path.isfile(p) else tree(p) for p in sorted]


def makedir_with_warning(path: str, msg: str | None = None) -> None:
    if os.path.exists(path):
        if msg is None:
            print(f"Warning: {path} already exists. The output will be overwritten.")
        else:
            print(msg)
    else:
        os.makedirs(path)
