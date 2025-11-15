Contributing
============

Thank you for your interest in contributing to AnalysisG!

How to Contribute
-----------------

There are many ways to contribute to the AnalysisG framework:

* Report bugs and issues
* Suggest new features
* Improve documentation
* Submit code contributions
* Share usage examples

Reporting Issues
----------------

If you find a bug or have a feature request:

1. Check if the issue already exists in the GitHub issue tracker
2. Create a new issue with a clear description
3. Include relevant information:
   
   * AnalysisG version
   * Operating system
   * Steps to reproduce (for bugs)
   * Expected vs actual behavior

Code Contributions
------------------

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository
2. Clone your fork:

.. code-block:: bash

   git clone https://github.com/your-username/AnalysisG.git
   cd AnalysisG

3. Build the project:

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make

Code Style
~~~~~~~~~~

Follow the existing code style:

* Use consistent indentation
* Add comments for complex logic
* Document public APIs
* Follow C++17 standards

Documentation
~~~~~~~~~~~~~

Update documentation when:

* Adding new features
* Changing APIs
* Fixing bugs that affect usage

Documentation is located in:

* ``docs/doxygen/`` - Doxygen documentation
* ``docs/source/`` - Sphinx documentation

Testing
~~~~~~~

Ensure your changes don't break existing functionality:

1. Run existing tests
2. Add tests for new features
3. Verify documentation builds

Submitting Changes
~~~~~~~~~~~~~~~~~~

1. Create a feature branch:

.. code-block:: bash

   git checkout -b feature/my-feature

2. Make your changes
3. Commit with clear messages:

.. code-block:: bash

   git commit -m "Add feature: description"

4. Push to your fork:

.. code-block:: bash

   git push origin feature/my-feature

5. Create a pull request on GitHub

Documentation Contributions
---------------------------

Improving Documentation
~~~~~~~~~~~~~~~~~~~~~~~

Documentation improvements are always welcome:

* Fix typos and grammar
* Clarify explanations
* Add examples
* Improve organization

Building Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build documentation:

.. code-block:: bash

   cd docs
   doxygen ../Doxyfile
   make html

View the result in ``docs/_build/html/index.html``.

Code of Conduct
---------------

Be respectful and professional in all interactions. We aim to create a
welcoming and inclusive environment for all contributors.

Getting Help
------------

If you need help:

* Check the documentation
* Ask in GitHub discussions
* Open an issue with questions

License
-------

By contributing, you agree that your contributions will be licensed under
the same license as the project.

Recognition
-----------

Contributors will be acknowledged in the project. Significant contributions
may be recognized in release notes and documentation.

Contact
-------

For questions about contributing, please open an issue on GitHub or contact
the maintainers.

Thank You
---------

Your contributions help make AnalysisG better for everyone!
