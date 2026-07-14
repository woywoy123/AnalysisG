.. _exports_hdf5:

Exporting Data to HDF5
======================

To handle massive machine learning datasets efficiently, ``AnalysisG`` interfaces directly with the HDF5 C++ API. These mechanics are encapsulated within the ``io`` class inside ``src/AnalysisG/modules/io/cxx/hdf5.cxx`` and operate on specialized data structures (e.g., ``graph_t`` template exports and ``graph_hdf5_w``).

HDF5 I/O Manager
----------------
The ``io`` class serves as the central manager, abstracting away dataset initialization, dataspaces, and I/O streams:

- **Initialization**: ``io::start(filename, mode)`` opens or truncates an ``H5::H5File`` instance based on the read/write mode requested.
- **Dataset Discovery**: The method ``io::dataset_names()`` leverages ``H5Literate`` and custom C callbacks to recursively list all dataset names within the open file.

Writing Graph Structures
------------------------
When serializing complex event graphs or structured graph features, the ``io`` template system safely converts C++ memory into contiguous blocks compatible with HDF5:

1. **Compound Datatypes**: The system registers C++ structs like ``graph_hdf5_w`` as HDF5 compound datatypes via the C-API function ``hid_t member(graph_hdf5_w)``. This defines to the HDF5 engine how the internal layout (variables, types, bytes) of the C++ object looks.
2. **Dataset Creation**: Using ``io::dataset(set_name, type, length)``, a 1D ``H5::DataSpace`` is allocated natively based on the size of the incoming batch payload.
3. **Write Execution**: Using templated methods (like ``io::write<graph_hdf5_w>``), arrays are serialized directly through ``dataset->write(inpt->data(), pairs)``, efficiently flushing datasets of graphs in bulk.
