.. _cmake_cybuild:

CMake and Cython Build System (cybuild)
=======================================

The ``AnalysisG`` build system leverages a robust combination of CMake, ``scikit-build-core``, and Cython. Because the framework heavily utilizes PyTorch libtorch C++ API (ATen), ROOT IO, and PyCUDA, managing linking and compiler flags across Python and C++ boundaries requires strict automation.

The Build Architecture
----------------------

The build lifecycle begins when a user runs ``pip install .`` or ``python -m build``.

1. **scikit-build-core**: Configured in ``pyproject.toml``, this backend intercepts the build request and delegates it directly to CMake, bypassing traditional ``setuptools``.
2. **CMake Bootstrapping**: The root ``CMakeLists.txt`` enforces C++20 and probes the environment for CUDA compilers. If CUDA is found, it fetches the appropriate ``libtorch-cxx11-abi`` archive natively using ``FetchContent``.
3. **Cython Translation via cybuild**: Custom CMake macros translate Python-level ``.pyx`` code into C++ extensions, linking them against the core ``AnalysisG`` libraries (like ``ctools``).

The ``cybuild`` and ``cysub_build`` Macros
------------------------------------------

Inside ``src/AnalysisG/CMakeLists.txt``, the build system implements two critical CMake functions: ``cybuild`` and ``cysub_build``. These abstract away the immense complexity of generating Cython C++ code and linking it dynamically.

### 1. ``cybuild`` (Single Module Compilation)

This function takes a source path, output destination, target name, and link dependencies. It operates in three distinct phases.

**Phase A: Code Translation (Cython to C++)**

.. code-block:: cmake

    add_custom_command(OUTPUT c${name}.cpp DEPENDS 
        ${links}
        ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pxd 
        ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx 
    VERBATIM COMMAND Python::Interpreter -m cython 
        --capi-reexport-cincludes
        --no-docstrings
        --verbose 
        --output-file c${name}.cpp
        --cplus ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx
    )

Instead of relying on CMake's native `UseCython` (which can be flaky with complex C++20 dependencies), ``cybuild`` explicitly invokes the Python interpreter to execute the Cython compiler. 
Crucially, it passes ``--cplus`` to force C++ output (rather than C) and ``--capi-reexport-cincludes`` to ensure that the generated C++ headers correctly expose the internal API definitions.

**Phase B: Library Generation and Linking**

.. code-block:: cmake

    python_add_library(${name} MODULE c${name}.cpp WITH_SOABI)
    target_link_libraries(${name} PUBLIC c${name} ${links})
    target_link_libraries(${name} PRIVATE ctools)
    target_compile_options(${name} PRIVATE -fPIC)

Once the ``c*.cpp`` file is generated, CMake wraps it into a Python extension module via ``python_add_library(MODULE WITH_SOABI)``. This ensures the output shared object (`.so`) is correctly tagged with the Python version ABI (e.g., ``cp314-x86_64-linux-gnu.so``). 
It statically links the internal ``ctools`` library and applies ``-fPIC`` (Position Independent Code) which is mandatory for Python C extensions.

**Phase C: Installation**

.. code-block:: cmake

    install(TARGETS ${name} DESTINATION ${out})

The built target is injected directly into the designated python site-packages path during the ``scikit-build-core`` wheel generation phase.

### 2. ``cysub_build`` (Recursive Batch Compilation)

The ``cysub_build`` function is an iterative wrapper around the core ``cybuild`` logic. It is used to recursively build entire subdirectories of Cython modules (e.g., inside the ``pyc/`` directory).

.. code-block:: cmake

    file(GLOB_RECURSE PYX ${ANALYSISG_SOURCE_DIR}/${path}/*.pyx)
    file(GLOB_RECURSE PYD ${ANALYSISG_SOURCE_DIR}/${path}/*.pxd)
    
    foreach(cyx IN LISTS PYX)
        cmake_path(GET cyx STEM name_)
        # [ ... cybuild compilation and linking logic ... ]
    endforeach()

By globbing all ``.pyx`` and ``.pxd`` files, ``cysub_build`` dynamically identifies every Cython module in the target directory tree, assigns it a unique CMake target based on its stem name, and compiles them concurrently if ``ninja`` is used as the underlying build generator.
