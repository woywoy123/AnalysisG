C++ Private Member Variables Reference
=======================================

This document provides comprehensive documentation for all private member variables and methods in the C++ modules. Private members handle internal state management, caching, and implementation details not exposed to users.

Overview
--------

Private members in AnalysisG serve several purposes:

1. **Internal State Management**: Track processing state, caches, and temporary data
2. **Build Pipelines**: Internal methods for constructing analysis workflows
3. **Data Structures**: Maps and containers for efficient lookups
4. **Thread Safety**: Mutexes and synchronization primitives
5. **File I/O**: Internal file handles and iterators

Core Template Modules
---------------------

analysis
~~~~~~~~

**Header**: ``modules/analysis/include/AnalysisG/analysis.h``

The analysis class has extensive private infrastructure for orchestrating the entire analysis workflow.

**Private Build Methods**
^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods construct different phases of the analysis pipeline:

.. cpp:function:: void check_cache()
   
   Check if event/graph data is already cached on disk to avoid reprocessing.
   
   **Called by**: ``start()``
   
   **Purpose**: Populates ``in_cache`` map to skip redundant event building

.. cpp:function:: void build_project()
   
   Initialize the analysis project structure and output directories.
   
   **Called by**: ``start()``
   
   **Creates**: Output directory structure, initializes sampletracer

.. cpp:function:: void build_events()
   
   Build event objects from ROOT files using registered event templates.
   
   **Called by**: ``start()``
   
   **Process**:
   
   1. Reads ROOT files via ``io`` reader
   2. For each file, instantiates event templates
   3. Calls ``event_template::build()`` on each event
   4. Caches results if enabled

.. cpp:function:: void build_selections()
   
   Apply selection templates to filter events.
   
   **Called by**: ``start()``
   
   **Process**: Iterates through registered selections, applies filters

.. cpp:function:: void build_graphs()
   
   Construct graph objects from events for GNN processing.
   
   **Called by**: ``start()``
   
   **Process**:
   
   1. For each event, call registered graph templates
   2. Build adjacency matrices and node/edge features
   3. Cache graph data

.. cpp:function:: void build_model_session()
   
   Set up training sessions for registered models.
   
   **Called by**: ``start()``
   
   **Process**:
   
   1. Initialize optimizers from ``model_sessions``
   2. Create dataloaders with k-fold splits
   3. Spawn training threads

.. cpp:function:: void build_inference()
   
   Run inference on data using pre-trained models.
   
   **Called by**: ``start()``
   
   **Process**: Load model weights, process data, save predictions

.. cpp:function:: bool build_metric()
   
   Compute metrics on model outputs.
   
   :returns: True if metrics computed successfully
   
   **Called by**: ``start()``

.. cpp:function:: void build_metric_folds()
   
   Compute metrics across k-fold cross-validation splits.

.. cpp:function:: void build_dataloader(bool training)
   
   Configure dataloader for training or inference.
   
   :param training: If true, set up for training; if false, for inference

.. cpp:function:: void fetchtags()
   
   Fetch k-fold tags from metadata for cross-validation.
   
   **Purpose**: Populates ``tags`` with fold assignments

**Private Static Helper Methods**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: static int add_content(std::map<std::string, torch::Tensor*>* data, std::vector<variable_t>* content, int index, std::string prefx, TTree* tt = nullptr)
   
   Add ROOT TTree content to tensor map.
   
   :param data: Output tensor map
   :param content: Variables to extract
   :param index: Event index
   :param prefx: Branch name prefix
   :param tt: ROOT TTree pointer
   :returns: Number of variables added

.. cpp:function:: static void add_content(std::map<std::string, torch::Tensor*>* data, std::vector<std::vector<torch::Tensor>>* buff, torch::Tensor* edge, torch::Tensor* node, torch::Tensor* batch, std::vector<long> mask)
   
   Add graph data (edges, nodes) to tensor buffers.
   
   :param data: Tensor map
   :param buff: Buffer for batched tensors
   :param edge: Edge index tensor
   :param node: Node feature tensor
   :param batch: Batch assignment tensor
   :param mask: Masking for valid entries

.. cpp:function:: static void execution(model_template* mdx, model_settings_t mds, std::vector<graph_t*>* data, size_t* prg, std::string output, std::vector<variable_t>* content, std::string* msg)
   
   Execute model training/inference in separate thread.
   
   :param mdx: Model instance
   :param mds: Model settings
   :param data: Graph data for processing
   :param prg: Progress counter (updated atomically)
   :param output: Output path for results
   :param content: Variables to save
   :param msg: Status message (updated by thread)

.. cpp:function:: static void execution_metric(metric_t* mt, size_t* prg, std::string* msg)
   
   Execute metric computation in separate thread.
   
   :param mt: Metric data structure
   :param prg: Progress counter
   :param msg: Status message

.. cpp:function:: static void initialize_loop(optimizer* op, int k, model_template* model, optimizer_params_t* config, model_report** rep)
   
   Initialize training loop for k-fold.
   
   :param op: Optimizer instance
   :param k: Fold number
   :param model: Model to train
   :param config: Optimizer configuration
   :param rep: Output report structure

**Private Template Helper**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: template<typename g> void safe_clone(std::map<std::string, g*>* mp, g* in)
   
   Safely clone template object if not already in map.
   
   :param mp: Map to check/insert into
   :param in: Template object to clone
   
   **Purpose**: Avoid duplicate clones of same template

**Private State Variables**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Label Management**:

.. cpp:member:: std::map<std::string, std::string> file_labels
   
   Maps file paths to user-assigned labels.

.. cpp:member:: std::map<std::string, event_template*> event_labels
   
   Maps labels to event template instances.

.. cpp:member:: std::map<std::string, metric_template*> metric_names
   
   Maps metric names to metric template instances.

.. cpp:member:: std::map<std::string, selection_template*> selection_names
   
   Maps selection names to selection template instances.

.. cpp:member:: std::map<std::string, std::map<std::string, graph_template*>> graph_labels
   
   Two-level map: event label → graph label → graph template.
   
   **Purpose**: Each event type can have multiple graph representations

**Model Management**:

.. cpp:member:: std::vector<std::string> model_session_names
   
   Names of all registered training sessions.

.. cpp:member:: std::map<std::string, model_template*> model_inference
   
   Models registered for inference (no training).

.. cpp:member:: std::map<std::string, model_template*> model_metrics
   
   Models to compute metrics on.

.. cpp:member:: std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions
   
   Training sessions: (model, optimizer config) pairs.

**Training Infrastructure**:

.. cpp:member:: std::map<std::string, optimizer*> trainer
   
   Active optimizer instances for each model.

.. cpp:member:: std::map<std::string, model_report*> reports
   
   Training reports (loss curves, metrics) for each session.

.. cpp:member:: std::vector<std::thread*> threads
   
   Worker threads for parallel training/inference.

**Cache Management**:

.. cpp:member:: std::map<std::string, std::map<std::string, bool>> in_cache
   
   Cache status: file → event_label → is_cached.
   
   **Purpose**: Skip processing if data already on disk

.. cpp:member:: std::map<std::string, bool> skip_event_build
   
   Whether to skip event building for each file.

.. cpp:member:: std::map<std::string, std::string> graph_types
   
   Maps graph labels to their type names.

**Core Components**:

.. cpp:member:: std::vector<folds_t>* tags
   
   K-fold cross-validation tags (train/val/test assignments).

.. cpp:member:: dataloader* loader
   
   Dataloader instance for batch iteration.

.. cpp:member:: sampletracer* tracer
   
   Tracks sample metadata (cross-sections, sum of weights).

.. cpp:member:: io* reader
   
   I/O handler for ROOT/HDF5 files.

.. cpp:member:: bool started
   
   Flag indicating if ``start()`` has been called.
   
   **Purpose**: Prevent double-execution

event_template
~~~~~~~~~~~~~~

**Header**: ``modules/event/include/templates/event_template.h``

**Private Methods**:

.. cpp:function:: void build_mapping(std::map<std::string, data_t*>* evnt)
   
   Build internal mapping between ROOT branches and particle objects.
   
   :param evnt: Event data from ROOT file
   
   **Process**:
   
   1. Parse branch names from ``evnt``
   2. Match to particle templates via ``add_leaf()`` mappings
   3. Populate ``tree_variable_link``

.. cpp:function:: void flush_leaf_string()
   
   Clear leaf string caches after event building.
   
   **Purpose**: Free memory from temporary branch name storage

**Private Variables**:

.. cpp:member:: std::map<std::string, bool> next_
   
   Tracks which trees have more entries to read.
   
   **Keys**: Tree names
   **Values**: Has next entry

.. cpp:member:: std::map<std::string, particle_template*> particle_generators
   
   Template particles used to create actual particle instances.
   
   **Purpose**: Factory pattern - clone generators for each event

.. cpp:member:: std::map<std::string, std::map<std::string, element_t>> tree_variable_link
   
   Links ROOT tree variables to internal element_t structures.
   
   **Structure**: tree_name → variable_name → element_t
   
   **Purpose**: Fast lookup during event building

.. cpp:member:: std::map<std::string, std::map<std::string, particle_template*>*> particle_link
   
   Links tree branches to particle collections.
   
   **Structure**: tree_name → branch_name → particle_map
   
   **Purpose**: Organize particles by their source tree/branch

.. cpp:member:: std::map<std::string, particle_template*> garbage
   
   Temporary particles to be deleted after event processing.
   
   **Purpose**: Memory management for cloned particles

graph_template
~~~~~~~~~~~~~~

**Header**: ``modules/graph/include/templates/graph_template.h``

**Private Variables**:

.. cpp:member:: bool is_owner
   
   Whether this graph owns its data (vs. referencing external data).
   
   **Purpose**: Prevent double-free of tensors

.. cpp:member:: std::mutex mut
   
   Mutex for thread-safe access to graph data.
   
   **Usage**: Lock when modifying edge_index or node features

.. cpp:member:: torch::Tensor* edge_index
   
   Pointer to edge connectivity tensor [2, num_edges].
   
   **Format**: Row 0 = source nodes, Row 1 = destination nodes

.. cpp:member:: std::map<std::string, int>* data_map_graph
   
   Maps graph-level feature names to column indices.

.. cpp:member:: std::map<std::string, int>* data_map_node
   
   Maps node feature names to column indices.

.. cpp:member:: std::map<std::string, int>* data_map_edge
   
   Maps edge feature names to column indices.

Data Management Modules
-----------------------

io
~~

**Header**: ``modules/io/include/io/io.h``

**Private HDF5 Methods**:

.. cpp:function:: hid_t member(folds_t t)
   
   Create HDF5 compound type for folds_t struct.
   
   :returns: HDF5 type identifier

.. cpp:function:: hid_t member(graph_hdf5_w t)
   
   Create HDF5 compound type for graph_hdf5_w struct.

.. cpp:function:: static herr_t file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata)
   
   HDF5 callback for iterating through datasets.
   
   :param loc_id: HDF5 location identifier
   :param name: Dataset/group name
   :param linfo: Link info
   :param opdata: User data
   :returns: HDF5 error code

.. cpp:function:: H5::DataSet* dataset(std::string set_name, hid_t type, long long unsigned int length)
   
   Create or open HDF5 dataset for writing.
   
   :param set_name: Dataset name
   :param type: HDF5 datatype
   :param length: Number of elements

.. cpp:function:: H5::DataSet* dataset(std::string set_name)
   
   Open existing HDF5 dataset for reading.

**Private ROOT Methods**:

.. cpp:function:: void root_key_paths(std::string path)
   
   Recursively scan ROOT file directory structure.
   
   :param path: Current directory path

.. cpp:function:: void root_key_paths(std::string path, TTree* t)
   
   Scan TTree branches.

.. cpp:function:: void root_key_paths(std::string path, TBranch* t)
   
   Scan TBranch leaves.

**Private Variables**:

**HDF5 State**:

.. cpp:member:: std::map<std::string, H5::DataSet*> data_w
   
   Open HDF5 datasets for writing.
   
   **Keys**: Dataset names

.. cpp:member:: std::map<std::string, H5::DataSet*> data_r
   
   Open HDF5 datasets for reading.

.. cpp:member:: H5::H5File* file
   
   Currently open HDF5 file handle.
   
   **Null when**: No file open

**ROOT State**:

.. cpp:member:: TFile* file_root
   
   Currently open ROOT file handle.

.. cpp:member:: std::map<std::string, data_t*>* iters
   
   Iterator state for ROOT tree reading.
   
   **Purpose**: Track current position in each tree

**Trigger Tracking**:

.. cpp:member:: std::map<std::string, bool> missing_trigger
   
   Tracks which triggers are missing from data.
   
   **Purpose**: Warn user about missing branches

.. cpp:member:: std::map<std::string, bool> success_trigger
   
   Tracks which triggers were successfully loaded.

dataloader
~~~~~~~~~~

**Header**: ``modules/dataloader/include/generators/dataloader.h``

**Private Members**:

.. cpp:member:: friend class analysis
   
   Grants analysis class access to private members.

.. cpp:member:: settings_t* setting
   
   Pointer to global analysis settings.
   
   **Purpose**: Access output paths, k-fold settings

.. cpp:member:: std::thread* cuda_mem
   
   Background thread for managing CUDA memory.
   
   **Purpose**: Asynchronous GPU memory allocation/deallocation

**Private Methods**:

.. cpp:function:: void cuda_memory_server()
   
   Worker function for CUDA memory management thread.
   
   **Process**:
   
   1. Monitor memory usage
   2. Allocate buffers as needed
   3. Clean up finished batches

.. cpp:function:: void clean_data_elements(std::map<std::string, int>** data_map, std::vector<std::map<std::string, int>*>* loader_map)
   
   Free data mapping structures.
   
   :param data_map: Data index maps to free
   :param loader_map: Loader maps to free

Analysis Modules
----------------

lossfx
~~~~~~

**Header**: ``modules/lossfx/include/templates/lossfx.h``

**Private Methods**:

.. cpp:function:: void interpret(std::string* ox)
   
   Parse optimizer name string.
   
   :param ox: Optimizer name to interpret

**Optimizer Builders** (one for each optimizer type):

.. cpp:function:: void build_adam(optimizer_params_t* op, std::vector<torch::Tensor>* params)
   
   Create Adam optimizer instance.

.. cpp:function:: void build_adagrad(optimizer_params_t* op, std::vector<torch::Tensor>* params)
   
   Create Adagrad optimizer.

.. cpp:function:: void build_adamw(optimizer_params_t* op, std::vector<torch::Tensor>* params)
   
   Create AdamW optimizer (Adam with weight decay).

.. cpp:function:: void build_lbfgs(optimizer_params_t* op, std::vector<torch::Tensor>* params)
   
   Create L-BFGS optimizer (quasi-Newton method).

.. cpp:function:: void build_rmsprop(optimizer_params_t* op, std::vector<torch::Tensor>* params)
   
   Create RMSprop optimizer.

.. cpp:function:: void build_sgd(optimizer_params_t* op, std::vector<torch::Tensor>* params)
   
   Create SGD optimizer (stochastic gradient descent).

**Loss Function Builders**:

.. cpp:function:: void build_fx_loss(torch::nn::BCELossImpl* lossfx_)
   
   Build binary cross-entropy loss function.
   
   :param lossfx_: Loss function implementation

meta
~~~~

**Header**: ``modules/meta/include/meta/meta.h``

**Private Methods**:

.. cpp:function:: void compiler()
   
   Compile metadata from all sources.
   
   **Process**: Aggregate cross-sections, weights, fold assignments

.. cpp:function:: float parse_float(std::string key, TTree* tr)
   
   Parse float value from ROOT TTree.
   
   :param key: Branch name
   :param tr: TTree pointer
   :returns: Float value

.. cpp:function:: std::string parse_string(std::string key, TTree* tr)
   
   Parse string value from ROOT TTree.

.. cpp:function:: static void get_isMC(bool*, meta*)
   
   Determine if sample is Monte Carlo or data.

.. cpp:function:: static void get_found(bool*, meta*)
   
   Check if metadata was successfully loaded.

**Private Variables**:

.. cpp:member:: std::vector<folds_t>* folds
   
   K-fold cross-validation assignments.
   
   **Structure**: Vector of fold structures with train/val/test indices

metric_template
~~~~~~~~~~~~~~~

**Header**: ``modules/metric/include/templates/metric_template.h``

**Private Variables**:

.. cpp:member:: friend class metric_template
   
   Self-friend for template specialization access.

.. cpp:member:: friend class analysis
   
   Grants analysis access to build metrics.

.. cpp:member:: mode_enum train_mode
   
   Current mode: TRAIN, VALIDATION, or TEST.

.. cpp:member:: std::string* pth
   
   Pointer to output path for metric results.

.. cpp:member:: model_template* mdlx
   
   Model being evaluated.

.. cpp:member:: metric_template* mtx
   
   Metric template instance.

.. cpp:member:: size_t index
   
   Current fold index.

**Private Methods**:

.. cpp:function:: void build()
   
   Build metric computation pipeline.

Common Patterns
---------------

Friend Classes
~~~~~~~~~~~~~~

Many modules declare ``friend class analysis`` to allow the orchestrator access to private members:

.. code-block:: cpp

   class my_module {
   private:
       friend class analysis;  // analysis can access private members
       // ...
   };

**Purpose**: analysis needs to coordinate between modules without exposing implementation details to users.

Cache Maps
~~~~~~~~~~

Pattern for tracking cached data:

.. code-block:: cpp

   std::map<std::string, std::map<std::string, bool>> in_cache;
   // file_path -> event_label -> is_cached

**Usage**: Check cache before expensive operations.

Data Index Maps
~~~~~~~~~~~~~~~

Pattern for fast feature lookup:

.. code-block:: cpp

   std::map<std::string, int>* data_map_node;
   // feature_name -> column_index

**Purpose**: Convert feature names to tensor column indices.

Thread Safety
~~~~~~~~~~~~~

Pattern for thread-safe data access:

.. code-block:: cpp

   class graph_template {
   private:
       std::mutex mut;
       torch::Tensor* edge_index;
       
       void modify_graph() {
           std::lock_guard<std::mutex> lock(mut);
           // Modify edge_index safely
       }
   };

Memory Management
~~~~~~~~~~~~~~~~~

Pattern for owning vs. referencing data:

.. code-block:: cpp

   class graph_template {
   private:
       bool is_owner;
       torch::Tensor* data;
       
       ~graph_template() {
           if (is_owner) {
               delete data;  // Only free if we own it
           }
       }
   };

See Also
--------

* :doc:`cpp_complete_reference`: Public API documentation
* :doc:`../core/overview`: Core package overview
* :doc:`../technical/cpp_modules_api`: Detailed module documentation
