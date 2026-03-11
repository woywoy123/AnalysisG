IO (Python)
===========

The ``IO`` Cython class wraps the C++ ROOT/HDF5 I/O engine.  It iterates
over events stored in ROOT TTree files and exposes them as dictionaries of
``data_t`` objects keyed by leaf name.

Constructor
-----------

``IO(root=[])`` — pass a list of ROOT file paths to open immediately.

Iteration Protocol
------------------

``IO`` implements Python's iterator protocol.  Each ``next()`` call returns
a dictionary:

.. code-block:: python

   reader = IO()
   reader.Trees   = ["nominal"]
   reader.Branches = ["jet_pt", "jet_eta"]
   reader.ScanKeys()
   reader.begin()
   for event in reader:
       jet_pt = event[b"nominal.jet_pt.jet_pt"]

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``ScanKeys()``
     - Scan the registered files and build the internal key map. Must be
       called before iterating.
   * - ``begin()``
     - Rewind the internal cursor to the first event.
   * - ``end()``
     - Release internal file handles and free memory.
   * - ``MetaData() → dict[str, Meta]``
     - Return a dictionary of :class:`Meta` objects (ATLAS dataset
       metadata) keyed by dataset label.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Type
     - Description
   * - ``Verbose``
     - ``bool``
     - Enable/disable console progress output.
   * - ``EnablePyAMI``
     - ``bool``
     - If ``True``, query the ATLAS AMI database for metadata.
   * - ``MetaCachePath``
     - ``str``
     - Directory where AMI metadata cache JSON files are stored.
   * - ``SumOfWeightsTreeName``
     - ``str``
     - Name of the ROOT tree that stores sum-of-weights information.
   * - ``Trees``
     - ``list[str]``
     - TTree names to read.
   * - ``Branches``
     - ``list[str]``
     - TBranch names to read.
   * - ``Leaves``
     - ``list[str]``
     - Registered leaf names (read-only; populated by ``ScanKeys``).
   * - ``Files``
     - ``list[str]``
     - List of ROOT file paths to process.
   * - ``Keys``
     - ``dict``
     - Nested dictionary ``{file: {tree: {branch: [leaves]}}}``.
