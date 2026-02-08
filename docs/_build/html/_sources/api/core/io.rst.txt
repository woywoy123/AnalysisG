I/O Module
==========

The I/O module provides a Cython interface to ROOT for reading n-tuples with minimal overhead.

Overview
--------

The ``io`` module (``src/AnalysisG/core/io.pyx``) provides simplified access to ROOT files:

- Read ROOT trees and branches
- Efficient data loading with caching
- Minimal syntax (can read ROOT files in 3 lines of code)
- Automatic type conversion between ROOT and Python
- Support for vector branches
- Friend tree support

Reading ROOT Files
------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import io
   
   # Open a ROOT file
   file = io.open("data.root")
   
   # Access a tree
   tree = file.Get("nominal")
   
   # Read branches
   jet_pt = tree.Get("jets_pt")
   jet_eta = tree.Get("jets_eta")

Reading Multiple Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import io
   import glob
   
   # Find all ROOT files in directory
   files = glob.glob("/data/ttbar/*.root")
   
   # Process each file
   for filepath in files:
       file = io.open(filepath)
       tree = file.Get("nominal")
       
       # Access branches
       data = tree.Get("variable_name")

Working with Vector Branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROOT vector branches (e.g., ``vector<float>``) are automatically converted:

.. code-block:: python

   from AnalysisG.core import io
   
   file = io.open("data.root")
   tree = file.Get("nominal")
   
   # Vector branch - returns list of lists
   jets_pt = tree.Get("jets_pt")  # [[jet1_pt, jet2_pt, ...], ...]
   
   # Access first event's jets
   event_0_jets = jets_pt[0]
   
   # Access specific jet
   event_0_jet_0_pt = jets_pt[0][0]

Friend Trees
~~~~~~~~~~~~

ROOT friend trees are supported for combining data from multiple trees:

.. code-block:: python

   from AnalysisG.core import io
   
   file = io.open("data.root")
   tree = file.Get("nominal")
   
   # Add friend tree
   tree.AddFriend("systematic", "syst.root")
   
   # Access branches from both trees
   nominal_pt = tree.Get("jets_pt")
   syst_pt = tree.Get("systematic.jets_pt")

API Reference
-------------

File Operations
~~~~~~~~~~~~~~~

.. function:: open(filepath: str)

   Open a ROOT file.

   :param filepath: Path to ROOT file
   :type filepath: str
   :return: ROOT file object
   
   Example:
   
   .. code-block:: python

      file = io.open("/data/sample.root")

Tree Operations
~~~~~~~~~~~~~~~

.. method:: File.Get(tree_name: str)

   Get a tree from the file.

   :param tree_name: Name of the tree
   :type tree_name: str
   :return: Tree object
   
   Example:
   
   .. code-block:: python

      tree = file.Get("nominal")

Branch Operations
~~~~~~~~~~~~~~~~~

.. method:: Tree.Get(branch_name: str)

   Read a branch from the tree.

   :param branch_name: Name of the branch
   :type branch_name: str
   :return: Branch data (list or nested list for vectors)
   
   Example:
   
   .. code-block:: python

      data = tree.Get("jets_pt")

.. method:: Tree.GetEntries() -> int

   Get the number of entries in the tree.

   :return: Number of entries
   :rtype: int

.. method:: Tree.GetEntry(index: int)

   Load a specific entry.

   :param index: Entry index
   :type index: int
   :return: Number of bytes read

.. method:: Tree.GetListOfBranches()

   Get list of all branches in the tree.

   :return: List of branch names
   :rtype: list

Advanced Features
-----------------

Caching
~~~~~~~

The I/O module implements intelligent caching:

- Branch data is cached after first access
- Reduces disk I/O for repeated access
- Automatic memory management

Type Conversion
~~~~~~~~~~~~~~~

Automatic conversion between ROOT and Python types:

- ``Float_t``, ``Double_t`` → ``float``
- ``Int_t``, ``Long64_t`` → ``int``
- ``Bool_t`` → ``bool``
- ``vector<T>`` → ``list``
- ``string`` → ``str``

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

The I/O module is optimized for:

- Minimal memory copying
- Efficient type conversion
- Lazy loading (only load requested branches)
- Parallel file reading (when using Analysis class)

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import io
   
   try:
       file = io.open("data.root")
   except FileNotFoundError:
       print("File not found")
   
   try:
       tree = file.Get("nonexistent_tree")
   except KeyError:
       print("Tree not found")

Integration with Analysis
--------------------------

The I/O module integrates seamlessly with the Analysis framework:

.. code-block:: python

   from AnalysisG.core import Analysis, EventTemplate
   
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           
           # Specify what to read
           self.Tree = "nominal"
           self.Branches = ["jets_pt", "jets_eta", "met_met"]
   
   # Analysis handles I/O automatically
   ana = Analysis()
   ana.AddSamples("/data/*.root", "signal")
   ana.AddEvent(MyEvent(), "events")
   ana.Start()  # I/O handled internally

Implementation Details
----------------------

C++ Backend
~~~~~~~~~~~

The I/O module wraps ROOT's C++ API:

- Uses ``TFile``, ``TTree``, ``TBranch`` classes
- Implements efficient buffer management
- Handles memory ownership correctly

Cython Interface
~~~~~~~~~~~~~~~~

The Cython implementation provides:

- Python-friendly API
- Automatic GIL management
- Proper exception handling
- Memory safety

See Also
--------

* :doc:`analysis` - Analysis class
* :doc:`templates` - Template classes
* `ROOT Documentation <https://root.cern/doc/master/>`_
