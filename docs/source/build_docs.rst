Building documentation
======================

Local build
-----------

From the repository root:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   doxygen docs/Doxyfile
   sphinx-build -b html docs/source docs/build/html

Local system prerequisites:

- ``doxygen`` (for C++ XML generation)
- ``graphviz`` (for inheritance diagrams)

This generates:

- Doxygen XML in ``docs/build/doxygen/xml``
- Sphinx HTML in ``docs/build/html``

Read the Docs configuration
---------------------------

Read the Docs uses ``.readthedocs.yaml`` with:

- Ubuntu 22.04
- Python 3.11
- ``doxygen`` and ``graphviz`` apt packages
- a pre-build step: ``doxygen docs/Doxyfile``
- Sphinx config at ``docs/source/conf.py``
- Python dependencies from ``docs/requirements.txt``
