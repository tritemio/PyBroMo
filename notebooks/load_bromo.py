#%%writefile load_bromo.py

ip = get_ipython()
if not 'NOTEBOOK_DIR' in globals():
    NOTEBOOK_DIR = ip.magic('%pwd')
    
from IPython.display import clear_output
from glob import glob

ip.magic('%matplotlib inline')
ip.magic('%cd ..')
BROWN_DIR = ip.magic('%pwd')
ip.magic('run brownian')

from utils import git

# If git is available, check PyBroMo version
if not git.git_path_valid():
    print('\nSoftware revision unknown (git not found).')
else:
    last_commit = git.get_last_commit_line()
    print('\nCurrent software revision:\n {}\n'.format(last_commit))
    if not git.check_clean_status():
        print('\nWARNING -> Uncommitted changes:')
        print(git.get_status())


