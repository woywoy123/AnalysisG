Missing CPU and CUDA Versions
=============================

The `pyc` module is built using CMake and Cython, explicitly configured to compile custom PyTorch extensions with native C++ and CUDA support.

CMake Configuration (CUDA Check)
--------------------------------
When `scikit-build-core` triggers the CMake compilation step, the `CMakeLists.txt` explicitly checks for the CUDA language module:

.. code-block:: cmake

    check_language(CUDA)
    enable_language(CUDA)

Fallback Behavior
-----------------
Currently, if a CUDA compiler (``nvcc``) is not found on the system during installation, the CMake configuration will fail unless explicitly bypassed. The PyCUDA (`pyc`) extensions in this framework are heavily optimized for GPU execution (using libtorch with `cu126`), meaning that running graph neural network clustering and physics transforms locally on a CPU requires a modified build environment.

If you intentionally want to run without CUDA:
1. You must disable the `CUDA` language requirement in the `pyc/CMakeLists.txt` project definition.
2. The PyTorch dependencies must be switched from `libtorch-cxx11-abi-shared-with-deps-2.7.0+cu126` to the CPU-only LibTorch equivalent.

*Note: Future updates will introduce an automatic CPU fallback flag directly in `pyproject.toml`.*
