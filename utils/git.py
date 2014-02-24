"""
Functions to check the version of the software by quering git.
"""

from subprocess import check_output, call
import os

from path_def import GIT_PATH


def git_path_valid(git_path=None):
    """
    Check whether git executable is found.
    """
    if git_path is None: git_path = GIT_PATH
    try:
        call([git_path, '--version'])
        return True
    except OSError:
        return False

def get_git_version(git_path=None):
    """
    Get the version string from Git executables.
    """
    if git_path is None: git_path = GIT_PATH
    git_version = check_output([git_path, "--version"]).split()[2]
    return git_version

def get_status(git_path=None):
    """
    Returns a string listing all the uncommitted changes in the working dir.
    """
    if git_path is None: git_path = GIT_PATH
    output = check_output([git_path, "status", "--porcelain"])
    return output

def check_clean_status(git_path=None):
    """
    Returns whether there are uncommitted changes in the working dir.
    """
    output = get_status(git_path)
    is_unmodified = (len(output.strip()) == 0)
    return is_unmodified

def get_last_commit_line(git_path=None):
    """
    Get one-line description of HEAD commit for repository in current dir.
    """
    if git_path is None: git_path = GIT_PATH
    output = check_output([git_path, "log", "--oneline", "-n1"])
    return output.strip()

def get_last_commit(git_path=None):
    """
    Get the HEAD commit SHA1 of repository in current dir.
    """
    if git_path is None: git_path = GIT_PATH
    line = get_last_commit_line(git_path)
    revision_id = line.split()[0]
    return revision_id
