============
Contributing
============

Thank you for your interest in contributing to AnalysisG! This guide will help you get started.

Getting Started
===============

Setting Up Development Environment
----------------------------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/AnalysisG.git
       cd AnalysisG

3. Create a branch for your changes:

   .. code-block:: bash

       git checkout -b feature/my-feature

4. Build the project:

   .. code-block:: bash

       mkdir build && cd build
       cmake ..
       make -j$(nproc)

Code Style
==========

C++ Guidelines
--------------

- Use C++17 features appropriately
- Follow existing code formatting
- Use meaningful variable and function names
- Document public interfaces with Doxygen comments

Example documentation style:

.. code-block:: cpp

    /**
     * @brief Brief description of the function.
     * 
     * Detailed description if needed.
     * 
     * @param param1 Description of first parameter.
     * @param param2 Description of second parameter.
     * @return Description of return value.
     */
    int myFunction(int param1, std::string param2);

Python Guidelines
-----------------

- Follow PEP 8 style guide
- Use type hints where appropriate
- Document with docstrings

Submitting Changes
==================

1. Commit your changes with clear messages:

   .. code-block:: bash

       git commit -m "Add feature X that does Y"

2. Push to your fork:

   .. code-block:: bash

       git push origin feature/my-feature

3. Open a Pull Request on GitHub

4. Describe your changes in the PR description

Testing
=======

Running Tests
-------------

Run the test suite before submitting:

.. code-block:: bash

    cd build
    make test

Adding Tests
------------

When adding new features, please include corresponding tests.

Documentation
=============

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

    cd docs
    doxygen Doxyfile
    pip install -r requirements.txt
    sphinx-build -b html . _build/html

Open ``_build/html/index.html`` in your browser to view.

Questions?
==========

If you have questions, feel free to:

- Open an issue on GitHub
- Check existing documentation
- Review similar PRs

We appreciate your contributions!
