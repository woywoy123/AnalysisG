Data Loader / Generator
=======================

The ``dataloader`` class inherits from ``notification`` and ``tools`` and
manages batched data delivery to GNN training loops.

It is responsible for k-fold dataset splitting, batch construction
(padding, device transfer, batching ``graph_t`` objects), HDF5
graph-cache persistence, and the inference data pipeline.

Key methods:

- ``get_k_train_set(k)`` — training partition for fold ``k``
- ``get_k_validation_set(k)`` — validation partition for fold ``k``
- ``get_test_set()`` — held-out evaluation set
- ``build_batch(data, mdl, rep)`` — collate a list of ``graph_t*`` into a single batched graph
- ``safe_delete(data)`` — deallocate a batch without leaking tensors
- ``generate_test_set(percentage)`` — split off ``percentage`` % of data as evaluation set
- ``generate_kfold_set(k)`` — generate k-fold splits
- ``dump_dataset(path)`` / ``restore_dataset(path)`` — serialise/deserialise k-fold splits
- ``dump_graphs(path, threads)`` — write all ``graph_t`` objects to HDF5
- ``restore_graphs(path, threads)`` — read ``graph_t`` objects from HDF5 cache
- ``datatransfer(op)`` — transfer all tensors to the device specified in ``op``
- ``get_inference()`` — retrieve inference data keyed by sample label
- ``start_cuda_server()`` — launch the background CUDA memory manager thread

.. doxygenclass:: dataloader
   :project: AnalysisG
   :members:
   :protected-members:
   :undoc-members:
