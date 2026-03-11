Installation
============

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Notes
   * - C++ compiler
     - C++20-capable (GCC ≥ 11, Clang ≥ 14).
   * - CMake
     - Version 3.15 or higher.
   * - Python
     - Version 3.8 or higher.
   * - Cython
     - ``pip install cython``
   * - HDF5
     - System package ``libhdf5-dev`` (Ubuntu) or equivalent.
   * - ROOT
     - A working installation from https://root.cern/.
   * - PyTorch / LibTorch
     - Installed automatically as part of the build process.
   * - RapidJSON
     - Fetched automatically by CMake.
   * - Graphviz *(optional)*
     - Required only for building the Doxygen class-diagram graphs.

Building from Source
--------------------

.. code-block:: bash

   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   pip install .

Building Documentation Locally
-------------------------------

Install the documentation dependencies::

   pip install -r docs/requirements.txt

Generate the Doxygen XML (from the repository root)::

   doxygen Doxyfile

Build the Sphinx HTML output (from the repository root)::

   python -m sphinx -b html docs/source docs/_build/html

The resulting HTML pages are written to ``docs/_build/html/``.
Open ``docs/_build/html/index.html`` in a browser to review the output.

.. tip::

   If doxygen is not installed you can still build the documentation.
   The Sphinx configuration detects the missing XML and disables the
   Breathe extension automatically. API reference pages will be empty
   but all other content renders normally.

Read the Docs
-------------

The documentation is hosted at `Read the Docs <https://readthedocs.org>`_.
It is rebuilt automatically on every push to the default branch.
The build pipeline is defined in ``.readthedocs.yaml`` at the repository root:

1. Doxygen is executed to produce XML in ``doxygen-docs/xml/``.
2. Sphinx (with the Breathe extension) reads that XML and renders HTML
   using the ``sphinx-rtd-theme``.
