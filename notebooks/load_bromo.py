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

