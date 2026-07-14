.. _interfaces_cython:

Cython Interfaces
=================

The ``AnalysisG`` project bridges complex C++ logic (such as neutrino kinematics, ROOT parsing, and graph operations) into Python by using Cython extensions compiled via scikit-build-core.

C++ and Python Interoperability
-------------------------------
The core C++ interfaces reside in the ``src/AnalysisG/pyc/interface`` directory. These routines focus heavily on data translation and offloading computation:

- **Data Translation**: Specialized routines like ``pyc::std_to_dict`` convert heavily nested C++ constructs (like ``std::map<std::string, torch::Tensor>``) into PyTorch dictionary types (``torch::Dict``) accessible directly in Python. 
- **Tensor Casting**: Standard C++ arrays and vectors (``std::vector<double>``, ``std::vector<long>``) are dynamically wrapped or cast into contiguous ``torch::Tensor`` objects through functions like ``pyc::tensorize``.
- **Device Management**: Operations are transparently offloaded to GPUs via LibTorch operations by leveraging routines such as ``changedev(dev, &tensor)`` internally.

Cython Implementation
---------------------
To integrate seamlessly, Cython (``.pyx`` and ``.pxd``) files define the bindings connecting the C++ signatures to Python functions:
- Custom ``.pxd`` declarations explicitly match the C++ headers.
- When compiling, the Cython engine is configured with ``--cplus`` to automatically translate these bindings into C++ boilerplate.
- Data pipelines (like PyTorch tensor manipulation and the ``nusol`` neutrino solver) invoke these bindings for extreme performance without GIL limitations.
