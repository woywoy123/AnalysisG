Core Modules and Languages used by AnalysisG
--------------------------------------------

To ensure optimal performance, the package uses C++ as the underlying language, but interfaces with Python using Cython.
Cython naturally interfaces with Python and provides minimal overhead in terms of multithreading limitations as would be the case of purely written Python code. 
Furthermore, Cython permits the sharing of C++ classes and pointers, meaning that memory is not unintentionally copied and introducing inefficiencies.

AnalysisG provides the following core modules that can be used in native C++, Cython and Python:

- **EventTemplate**: A template class used to specify to the framework which type of event and particle definitions to be used for the event.
- **ParticleTemplate**: A template class used in conjunction with **EventTemplate** to define the underlying particle type.
- **GraphTemplate**: A template class used to define the inclusive graph features, such as edge, node and global graph attributes. 
- **SelectionTemplate**: A template class for defining a customized event selection algorithm. 
- **Plotting**: A wrapper around **boost_histograms** and **mpl-hepp** that uses an object like syntax to define plotting routines.
- **io**: A Cython interface for the CERN ROOT C++ package, which centers around being simple to use and requiring as minimal syntax as possible to read ROOT n-tuples.
- **ModelTemplate**: A template class used to define machine learning algorithms.
- **Analysis**: The main analysis compiler used to define chains of actions to be performed given a user defined template class.
- **Tools**: A collection of tools that are used by the package.
- **pyc (Python CUDA)**: A completely detached package which implements high performance native C++ and CUDA code/kernels, utilizing the **PyTorch** API. 


