Data Loader / Generator
========================

The ``dataloader`` class manages the in-memory graph dataset, providing k-fold
splitting, test-set generation, HDF5-based graph caching, and batching for
multi-GPU training.  It is populated by ``sampletracer::populate_dataloader``
and used exclusively by ``optimizer``.

Class: ``dataloader``
----------------------

**Header:** ``<generators/dataloader.h>``

**Inheritance:** ``notification``, ``tools``

Dataset Management Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``void generate_test_set(float percentage = 50)``
     - Reserves *percentage* percent of the dataset as the held-out test set.
       The remainder is used for k-fold training/validation.
   * - ``void generate_kfold_set(int k)``
     - Partitions the training set into *k* folds.
   * - ``std::vector<graph_t*>* get_k_train_set(int k)``
     - Returns the training subset for fold *k*.
   * - ``std::vector<graph_t*>* get_k_validation_set(int k)``
     - Returns the validation subset for fold *k*.
   * - ``std::vector<graph_t*>* get_test_set()``
     - Returns the held-out test set.
   * - ``std::map<std::string, std::vector<graph_t*>>* get_inference()``
     - Returns the inference dataset (label → graphs).

Batching Methods
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Signature
     - Description
   * - ``std::vector<graph_t*>* build_batch(std::vector<graph_t*>* data, model_template* mdl, model_report* rep)``
     - Builds a batched ``graph_t`` by concatenating the tensors of all
       graphs in *data* along the batch dimension.
   * - ``static void safe_delete(std::vector<graph_t*>* data)``
     - Deletes batched ``graph_t`` objects that were created by ``build_batch``.
   * - ``std::vector<graph_t*> get_random(int num = 5)``
     - Returns *num* randomly sampled ``graph_t*`` objects from the dataset.
   * - ``void extract_data(graph_t* gr)``
     - Moves the tensor data of *gr* to CPU memory (for serialisation).

Device Transfer Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``void datatransfer(torch::TensorOptions* op, size_t* num_events, size_t* prg_events)``
     - Transfers the entire dataset to the device specified by *op*.
   * - ``void datatransfer(std::map<int, torch::TensorOptions*>* ops)``
     - Transfers to multiple devices simultaneously (one per kfold/GPU).
   * - ``void start_cuda_server()``
     - Starts the background CUDA memory management thread (CUDA builds only).

Cache / Restore Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``bool dump_graphs(std::string path = "./", int threads = 10)``
     - Serialises all graphs to HDF5 files in *path* using *threads* workers.
       Returns ``false`` on failure.
   * - ``void restore_graphs(std::vector<std::string> paths, int threads)``
     - Deserialises graphs from HDF5 files at *paths*.
   * - ``void restore_graphs(std::string paths, int threads)``
     - Deserialises graphs from all HDF5 files in directory *paths*.
   * - ``void dump_dataset(std::string path)``
     - Dumps the k-fold train/validation/test split indices to *path*.
   * - ``bool restore_dataset(std::string path)``
     - Restores the split indices from *path*.  Returns ``false`` on failure.
