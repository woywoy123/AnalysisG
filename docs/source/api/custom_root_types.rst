.. _custom-root-types:

Adding New Data Types to ROOT I/O
=================================

Integrating new data types into AnalysisG's ROOT I/O pipeline requires mapping those types across both the ROOT interface (e.g., ``TTree``, ``TBranch``, ``TLeaf``) and the internal representation blocks (e.g., ``data_t`` mappings).

While AnalysisG automates much of the tree unrolling (as seen in ``io::root_key_paths`` inside ``root.cxx``), you may need to extend or adapt types when introducing custom structs or completely unsupported types.

How ROOT I/O parses Types
-------------------------
The ``io`` module recursively scans the keys of the given ROOT files. When encountering a ``TTree``, it looks up requested ``TBranch`` and ``TLeaf`` objects:

1. **Leaf Typing**: The type of a ``TLeaf`` is fetched via ``_lf->GetTypeName()`` and cached in ``leaf_typed``.
2. **Data Handlers**: A ``data_t`` handler is instantiated for each valid leaf to store its path, tree name, leaf name, and its type.
3. **Buffer Mapping**: The initialization and parsing logic link the file contents directly to C++ memory types based on the detected ``GetTypeName()``.

HDF5 Hybridization
------------------
If your new data types are going to be serialized out to HDF5 (for GNN training or caching), you must also implement equivalent HDF5 definitions.
For example, in ``src/AnalysisG/modules/io/cxx/types.cxx``, new structures (like ``folds_t`` or ``graph_hdf5_w``) are mapped to ``hid_t`` compound types:

.. code-block:: cpp

    hid_t px = H5Tcreate(H5T_COMPOUND, sizeof(my_custom_struct)); 
    H5Tinsert(px, "field1", HOFFSET(my_custom_struct, field1), H5T_NATIVE_INT);
    // ...

Extending C++ Support
---------------------
When introducing custom C++ objects (like a new particle class) to be embedded directly inside a ROOT TTree:

- Ensure the object is defined in a header parsed by the PCM generator (see :ref:`ROOT to PCM Mapping <custom-root-pcm>`).
- If you're using basic scalar types or ``std::vector`` wrappers of basic types, AnalysisG handles them natively without additional structural mapping.
