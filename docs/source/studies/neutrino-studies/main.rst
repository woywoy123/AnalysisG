Analytical Neutrino Reconstruction Algorithm
============================================

This segment of the repository is used to understanding the analytical single and double neutrino reconstruction algorithm presented in [1]. 
A complete reimplementation of the presented algorithm can be found under the *torch-extensions* directory, which uses the underlying codename **pyc**. 
The reimplementation aims to make the algorithm available in the context of machine learning, where speed is often necessary. 
Hence, multiple versions of it have been reimplemented, most notably is the native CUDA version. 

In order to assert the validity of the codebase, multiple comparative studies are shown between the original and the reimplementation.
This is followed by performing performance benchmarks between CPU and CUDA, and how unit scales impact the solutions of the algorithm.

Finally, a proposed improvement of the original algorithm is being illustrated.

.. toctree:: 
   :titlesonly:

   single-neutrino/main.rst
   double-neutrino/main.rst 


[1]. **Analytic solutions for neutrino momenta in decay of top quarks** (arXiv: 1305.1878v2)

