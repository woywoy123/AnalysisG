Installing Analysis-G
=====================
First clone the project into a target directory. 

.. code-block:: console
    :caption: Cloning the Project via Git

    git clone https://github.com/woywoy123/AnalysisG.git

To automate most of the setup, navigate to the **setup-scripts** directory and run the **setup-venv.sh** script.
This will generate a new Python environment called **GNN**, which can be sourced from the shell script **source_this.sh**. 

.. code-block:: console
    :caption: Running the Installation script 

    cd AnalysisG/setup-scripts && bash setup-venv.sh

If you are running the framework on a HPC cluster, for instance **lxplus** or some other Linux environment, make sure to have at least **GCC 6.20** enabled.
As an example, on **lxplus** machines, one can add the following to their bashrc file: 

.. code-block:: console
   
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
    source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
    lsetup "gcc gcc620_x86_64_slc6"

Or alternatively, run the installer as highlighted once and then create a **shell** alias which sources the framework's setup scripts. 
This could look something like this: 

.. code-block:: console 

   alias GNN='source <some path to repository>/setup-scripts/source_this.sh

So now one only has to type in **GNN** into their bash console once to have everything setup. 

Additional Software Setup
=========================
Analysis-G is dependent on one additional package, which can be found under **torch-extensions**.
This package directory is completely independent of the framework and contains several modules that contain physics specific functions. 
These are written in **C++ and CUDA**, and compiled using **CMake** using a Python build tool called **scikit-build-core**. 
Once compiled, modules will be available under the library name, **PyC**, which stands for **Python-C++/CUDA**.

