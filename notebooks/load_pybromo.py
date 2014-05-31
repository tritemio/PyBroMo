#%%writefile load_pybromo.py

ip = get_ipython()
if not 'NOTEBOOK_DIR' in globals():
    NOTEBOOK_DIR = ip.magic('%pwd')

from IPython.display import clear_output
from glob import glob

ip.magic('%matplotlib inline')
#ip.magic('%cd ~/src/pybromo')
ip.magic(r'%cd C:\Data\Antonio\software\src\pybromo/')
BROWN_DIR = ip.magic('%pwd')
ip.magic('run brownian')

from utils import git

# If git is available, check PyBroMo version
git.print_summary('PyBroMo')


