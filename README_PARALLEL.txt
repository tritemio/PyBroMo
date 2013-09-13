Official documentation:

http://ipython.org/ipython-doc/dev/parallel/parallel_process.html


Parallel computing on a single machine
======================================

Method 1
--------

Launch the notebook server and from the cluster tab start 4 engines

Method 2
--------
Open a terminal (cmd.exe) and type:

ipcluster start -n 4

NOTE: A scientific python distribution like Anaconda must be installed.


Parallel computing on many machines (Windows 7)
===============================================

Official docs:
http://ipython.org/ipython-doc/dev/parallel/parallel_process.html#starting-the-controller-and-engines-on-different-hosts

Here we configure 2 machines, one controller host that launch the simulation
and one "slave" host that performs the computation. This procedure can be extended
to multiple "slave" machine just repeating this same configuration.

Windows note
------------
It is assumed that a scientific python distribution like Anaconda is installed.
All the commands must be pasted in a cmd.exe terminal.

Setup the controller
--------------------

First time only, create an ipython profile.

    ipython profile create --parallel --profile=parallel

Now there is a new set of configuration files created in 
IPYTHONDIR/profile_parallel, where IPYTHONDIR is usually a folder named 
.ipython in the user home folder (C:\Users\username\). These files can be 
customized to change the default behaviour, if needed.

Now, each time we want to start a parallel computation we begin starting
the controller:

    ipcontroller --profile=parallel --ip=169.232.130.141
	
(where you have to specify the controller ip address)
	
This command creates a file ipcontroller-engine.json that contains
the connection info that the other machines need to connect to the controller.
The file is located in IPYTHONDIR/profile_parallel/security.

We need to copy ipcontroller-engine.json to the computation machine. This
can be done manually or you can link the file to a dropbox folder al it
will be automatically copied to hte other machines.

Setup the "slave" machine
-------------------------

Also on the machine in which we run the computation it's useful to create
a profile (only the first time), with the same command:

    ipython profile create --parallel --profile=parallel
	
as before, a set co new configuration files is created in the folder
IPYTHONDIR/profile_parallel.

We can start a computation engine with ipengine command, specifing the
path of the ipcontroller-engine.json file:

    ipengine --profile=parallel --file=C:\Data\Antonio\software\Dropbox\ipython\profile_parallel\security\ipcontroller-engine.json
	
or, we can write the file name in the configuration file so we don't need
to write it every time. To do so, edit the file ipengine_config.py
found in the previously created profile folder (IPYTHONDIR/profile_parallel).
Find the line:

    #c.IPEngineApp.url_file = u''
	
remove the trailing # and write the ipcontroller-engine.json path, in our
example:

    c.IPEngineApp.url_file = u'C:\Data\Antonio\software\Dropbox\ipython\profile_parallel\security\ipcontroller-engine.json'
	
Now to launch an engine simply type:

    ipengine --profile=parallel
	
It is suggested to launch as many engine as the number of cores. To launch
a second engine open a new terminal and type again the command, and so on.

To add another machine for computation just repeat the previous steps.

Launching the simulation
========================

Once the cluster is started (either in a single machine or on multiple 
machines) everithing is ready to launch a simulation.

On the controller machine start an IPython QtConsole or an 
IPython notebook using the profile "parallel":

    ipython qtconsole --profile=parallel
   
or 

    ipython notebook --profile=parallel

Then do:

from IPython.parallel import Client
rc = Client()
rc.ids

the last command shoud print the number of engines that were started.

Alternatively, if you have a qtconsole or notebook already started without
the profile parallel you can simply specify the path of the file
that contains the clients (engines) information. This file is 
ipcontroller-client.json (not -engines as before!) and is located in the 
profile folder.







