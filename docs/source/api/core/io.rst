I/O Module
==========

The I/O module provides a Cython interface to ROOT for reading n-tuples.

Overview
--------

The ``IO`` class (``src/AnalysisG/core/io.pyx``) provides simplified access to ROOT files:

- Read ROOT trees, branches and leaves
- Efficient data loading with iteration
- Automatic type conversion between ROOT and Python
- Support for vector branches
- PyAMI metadata integration

The IO Class
------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   # Create IO instance with file path(s)
   io = IO(["data.root"])
   
   # Or set files later
   io = IO()
   io.Files = ["data1.root", "data2.root"]
   
   # Specify trees and leaves to read
   io.Trees = ["nominal"]
   io.Leaves = ["jets_pt", "jets_eta", "met_met"]
   
   # Iterate over events
   for event in io:
       # Access data with byte keys: tree.branch.leaf
       pt = event[b"nominal.jets_pt.jets_pt"]
       met = event[b"nominal.met_met.met_met"]

Reading Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   io = IO(["sample.root"])
   io.Trees = ["nominal", "truth"]
   io.Leaves = ["weight_mc", "weight_pileup", "children_pt"]
   
   for event in io:
       # Access from different trees
       if b"truth.weight_mc.weight_mc" in event:
           truth_weight = event[b"truth.weight_mc.weight_mc"]
       
       if b"nominal.weight_mc.weight_mc" in event:
           nominal_weight = event[b"nominal.weight_mc.weight_mc"]

Specifying Branches
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   io = IO()
   io.Files = ["data.root"]
   io.Trees = ["nominal"]
   io.Branches = ["children_index"]  # Specify branches
   io.Leaves = ["met_met", "children_pt"]  # Specify leaves
   
   for event in io:
       # Access data
       data = event[b"nominal.children_pt.children_pt.children_pt"]

API Reference
-------------

Constructor
~~~~~~~~~~~

.. class:: IO(files=[])

   Create an IO instance.

   :param files: Optional list of ROOT file paths or single file path string
   :type files: list or str

Properties
~~~~~~~~~~

.. attribute:: IO.Files
   :type: list

   List of ROOT file paths to read.

.. attribute:: IO.Trees
   :type: list

   List of tree names to read from ROOT files.

.. attribute:: IO.Branches
   :type: list

   List of branch names to read.

.. attribute:: IO.Leaves  
   :type: list

   List of leaf names to read.

.. attribute:: IO.Verbose
   :type: bool

   Enable/disable verbose output. Default: True

.. attribute:: IO.EnablePyAMI
   :type: bool

   Enable PyAMI metadata fetching. Default: False

.. attribute:: IO.MetaCachePath
   :type: str

   Path to cache PyAMI metadata.

Methods
~~~~~~~

.. method:: IO.ScanKeys()

   Scan available keys in ROOT files and check for missing trees/branches/leaves.

.. method:: IO.MetaData() -> dict

   Get metadata for loaded files. Requires EnablePyAMI = True.

   :return: Dictionary mapping file paths to Meta objects
   :rtype: dict

.. method:: IO.Keys -> dict

   Get dictionary of available and missing keys in ROOT files.

   :return: Dictionary with "found" and "missed" keys for each file
   :rtype: dict

Iteration
~~~~~~~~~

The IO class is iterable:

.. code-block:: python

   io = IO(["data.root"])
   io.Trees = ["nominal"]
   io.Leaves = ["jets_pt"]
   
   for event in io:
       # event is a dictionary with byte keys
       print(event[b"nominal.jets_pt.jets_pt"])

Length
~~~~~~

Get number of events:

.. code-block:: python

   io = IO(["data.root"])
   io.Trees = ["nominal"]
   io.Leaves = ["weight_mc"]
   
   num_events = len(io)  # Total number of events

Data Access Keys
----------------

Data is accessed using byte strings with the format:

``tree_name.branch_name.leaf_name``

For vector branches (e.g., ``vector<float>``), additional ``.leaf_name`` may be appended:

``tree_name.branch_name.leaf_name.leaf_name``

Example:

.. code-block:: python

   # Simple leaf
   met = event[b"nominal.met_met.met_met"]
   
   # Vector branch
   jets_pt = event[b"nominal.children_pt.children_pt.children_pt"]
   # Result is a nested list: [[pt1, pt2, ...], ...]

Metadata Integration
--------------------

PyAMI Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   io = IO("./sample.root")
   io.MetaCachePath = "./meta_cache"
   io.Trees = ["nominal"]
   io.Leaves = ["weight_mc"]
   io.EnablePyAMI = True
   
   # Access metadata
   meta_dict = io.MetaData()
   meta = list(meta_dict.values())[0]
   
   # Available metadata fields
   print(meta.dsid)
   print(meta.crossSection)
   print(meta.genFiltEff)
   print(meta.totalEvents)
   print(meta.generators)
   print(meta.logicalDatasetName)

See Also
--------

* :doc:`analysis` - Analysis class
* :doc:`templates` - Template classes
* `ROOT Documentation <https://root.cern/doc/master/>`_
