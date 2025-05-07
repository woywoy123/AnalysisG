.. cpp:struct:: graph_t

    A data structure holding the exported graph information, including features, topology, and metadata.

    This struct acts as a container for graph data, separating truth and reconstructed information
    for graph, node, and edge levels. It also manages data transfer to different compute devices (CPU/GPU).

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_truth_graph(std::string name, g* mdl)

        Retrieves a truth-level graph feature tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param name: The name of the graph feature to retrieve (without the "T-" prefix).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the requested torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_truth_node(std::string name, g* mdl)

        Retrieves a truth-level node feature tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param name: The name of the node feature to retrieve (without the "T-" prefix).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the requested torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_truth_edge(std::string name, g* mdl)

        Retrieves a truth-level edge feature tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param name: The name of the edge feature to retrieve (without the "T-" prefix).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the requested torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_data_graph(std::string name, g* mdl)

        Retrieves a data-level (reconstructed) graph feature tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param name: The name of the graph feature to retrieve (without the "D-" prefix).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the requested torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_data_node(std::string name, g* mdl)

        Retrieves a data-level (reconstructed) node feature tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param name: The name of the node feature to retrieve (without the "D-" prefix).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the requested torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_data_edge(std::string name, g* mdl)

        Retrieves a data-level (reconstructed) edge feature tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param name: The name of the edge feature to retrieve (without the "D-" prefix).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the requested torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_edge_index(g* mdl)

        Retrieves the edge index tensor (topology) for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the edge index torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_event_weight(g* mdl)

        Retrieves the event weight tensor for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the event weight torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_batch_index(g* mdl)

        Retrieves the batch index tensor (mapping nodes to graphs in a batch) for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the batch index torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: template <typename g> \
                         torch::Tensor* get_batched_events(g* mdl)

        Retrieves the tensor indicating event boundaries within a batch for a specific model context.

        :tparam g: The type of the model or context requesting the data (used to determine the device).
        :param mdl: Pointer to the model or context object.
        :return: Pointer to the batched events torch::Tensor on the correct device, or nullptr if not found.

    .. cpp:function:: torch::Tensor* has_feature(graph_enum tp, std::string name, int dev)

        Checks for and retrieves a feature tensor based on its type, name, and target device.

        :param tp: The type of feature (e.g., data_node, truth_graph). See graph_enum.
        :param name: The name of the feature. Prefixes ("D-", "T-") are added internally based on 'tp'.
        :param dev: The index of the target device.
        :return: Pointer to the requested torch::Tensor on the specified device, or nullptr if not found.

    .. cpp:function:: void add_truth_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps)

        Adds truth-level graph features and their name-to-index mapping.

        :param data: A map where keys are feature names ("T-...") and values are pointers to the feature tensors.
        :param maps: A map where keys are feature names ("T-...") and values are their integer indices.

        .. note:: This function should only be called once during graph_t initialization.

    .. cpp:function:: void add_truth_node( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps)

        Adds truth-level node features and their name-to-index mapping.

        :param data: A map where keys are feature names ("T-...") and values are pointers to the feature tensors.
        :param maps: A map where keys are feature names ("T-...") and values are their integer indices.

        .. note:: This function should only be called once during graph_t initialization.

    .. cpp:function:: void add_truth_edge( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps)

        Adds truth-level edge features and their name-to-index mapping.

        :param data: A map where keys are feature names ("T-...") and values are pointers to the feature tensors.
        :param maps: A map where keys are feature names ("T-...") and values are their integer indices.

        .. note:: This function should only be called once during graph_t initialization.

    .. cpp:function:: void add_data_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps)

        Adds data-level (reconstructed) graph features and their name-to-index mapping.

        :param data: A map where keys are feature names ("D-...") and values are pointers to the feature tensors.
        :param maps: A map where keys are feature names ("D-...") and values are their integer indices.

        .. note:: This function should only be called once during graph_t initialization.

    .. cpp:function:: void add_data_node(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps)

        Adds data-level (reconstructed) node features and their name-to-index mapping.

        :param data: A map where keys are feature names ("D-...") and values are pointers to the feature tensors.
        :param maps: A map where keys are feature names ("D-...") and values are their integer indices.

        .. note:: This function should only be called once during graph_t initialization.

    .. cpp:function:: void add_data_edge(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps)

        Adds data-level (reconstructed) edge features and their name-to-index mapping.

        :param data: A map where keys are feature names ("D-...") and values are pointers to the feature tensors.
        :param maps: A map where keys are feature names ("D-...") and values are their integer indices.

        .. note:: This function should only be called once during graph_t initialization.

    .. cpp:function:: void transfer_to_device(torch::TensorOptions* dev)

        Transfers all tensor data (features, edge index, etc.) to the specified device.

        If the data is already on the target device, this operation is skipped.
        This operation is thread-safe using a mutex.

        :param dev: Pointer to torch::TensorOptions specifying the target device (e.g., torch::kCUDA:0).

    .. cpp:function:: void _purge_all()

        Deallocates all dynamically allocated memory within the graph_t object.

        This includes deleting tensor pointers, clearing maps and vectors, and deleting string pointers
        if this object is the owner (i.e., created via deserialization).

    .. cpp:member:: int num_nodes = 0
        Number of nodes in the graph.

    .. cpp:member:: long event_index = 0
        Index of the original event this graph corresponds to.

    .. cpp:member:: double event_weight = 1
        Weight associated with the event.

    .. cpp:member:: bool preselection = true
        Flag indicating if the event passed preselection criteria.

    .. cpp:member:: std::vector<long> batched_events = {}
        Stores indices defining event boundaries within a batch.

    .. cpp:member:: std::string* hash = nullptr
        Pointer to the unique hash string identifying the event.

    .. cpp:member:: std::string* filename = nullptr
        Pointer to the name of the file the event originated from.

    .. cpp:member:: std::string* graph_name = nullptr
        Pointer to the name assigned to this graph type.

    .. cpp:member:: c10::DeviceType device = c10::kCPU
        The primary device type where the data currently resides (initially CPU).

    .. cpp:member:: int in_use = 1
        Reference counter (potentially for managing lifetime in data loaders).

    // --- Private Members ---
    // Note: Sphinx C++ domain doesn't have explicit private markers like Doxygen.
    // Access control is implied by C++ rules. Friends are noted here.

    .. note::
        ``graph_template`` and ``dataloader`` are friend classes.

    .. cpp:member:: bool is_owner = false
        True if this object owns the memory for strings and maps (set during deserialization).

    .. cpp:member:: std::mutex mut
        Mutex to protect device transfer operations.

    .. cpp:member:: torch::Tensor* edge_index = nullptr
        Pointer to the edge index tensor [2, num_edges] on CPU.

    .. cpp:member:: std::map<std::string, int>* data_map_graph = nullptr
        Maps data graph feature names to indices.

    .. cpp:member:: std::map<std::string, int>* data_map_node  = nullptr
        Maps data node feature names to indices.

    .. cpp:member:: std::map<std::string, int>* data_map_edge  = nullptr
        Maps data edge feature names to indices.

    .. cpp:member:: std::map<std::string, int>* truth_map_graph = nullptr
        Maps truth graph feature names to indices.

    .. cpp:member:: std::map<std::string, int>* truth_map_node  = nullptr
        Maps truth node feature names to indices.

    .. cpp:member:: std::map<std::string, int>* truth_map_edge  = nullptr
        Maps truth edge feature names to indices.

    .. cpp:member:: std::vector<torch::Tensor*>* data_graph = nullptr
        Vector of data graph feature tensors on CPU.

    .. cpp:member:: std::vector<torch::Tensor*>* data_node  = nullptr
        Vector of data node feature tensors on CPU.

    .. cpp:member:: std::vector<torch::Tensor*>* data_edge  = nullptr
        Vector of data edge feature tensors on CPU.

    .. cpp:member:: std::vector<torch::Tensor*>* truth_graph = nullptr
        Vector of truth graph feature tensors on CPU.

    .. cpp:member:: std::vector<torch::Tensor*>* truth_node  = nullptr
        Vector of truth node feature tensors on CPU.

    .. cpp:member:: std::vector<torch::Tensor*>* truth_edge  = nullptr
        Vector of truth edge feature tensors on CPU.

    .. cpp:member:: std::map<int, std::vector<torch::Tensor>> dev_data_graph = {}
        Maps device index to a vector of data graph tensors for that device.

    .. cpp:member:: std::map<int, std::vector<torch::Tensor>> dev_data_node  = {}
        Maps device index to a vector of data node tensors for that device.

    .. cpp:member:: std::map<int, std::vector<torch::Tensor>> dev_data_edge  = {}
        Maps device index to a vector of data edge tensors for that device.

    .. cpp:member:: std::map<int, std::vector<torch::Tensor>> dev_truth_graph = {}
        Maps device index to a vector of truth graph tensors for that device.

    .. cpp:member:: std::map<int, std::vector<torch::Tensor>> dev_truth_node  = {}
        Maps device index to a vector of truth node tensors for that device.

    .. cpp:member:: std::map<int, std::vector<torch::Tensor>> dev_truth_edge  = {}
        Maps device index to a vector of truth edge tensors for that device.

    .. cpp:member:: std::map<int, torch::Tensor> dev_edge_index   = {}
        Maps device index to the edge index tensor for that device.

    .. cpp:member:: std::map<int, torch::Tensor> dev_batch_index  = {}
        Maps device index to the batch index tensor for that device.

    .. cpp:member:: std::map<int, torch::Tensor> dev_event_weight = {}
        Maps device index to the event weight tensor for that device.

    .. cpp:member:: std::map<int, torch::Tensor> dev_batched_events = {}
        Maps device index to the batched events tensor for that device.

    .. cpp:member:: std::map<int, bool> device_index = {}
        Tracks which devices the data has been transferred to.

    .. cpp:function:: void meta_serialize(std::map<std::string, int>* data, std::string* out)

        Serializes a map (string to int) into a single string representation.

        :param data: Pointer to the map to serialize.
        :param out: Pointer to the output string. Format: "key1|val1%key2|val2%..." or "NULL" if empty.

    .. cpp:function:: void meta_serialize(std::vector<torch::Tensor*>* data, std::string* out)

        Serializes a vector of tensor pointers using torch::pickle_save and base64 encoding.

        :param data: Pointer to the vector of tensor pointers to serialize.
        :param out: Pointer to the output base64 encoded string. "NULL" if empty.

    .. cpp:function:: void meta_serialize(torch::Tensor* data, std::string* out)

        Serializes a single tensor pointer using torch::pickle_save and base64 encoding.

        :param data: Pointer to the tensor to serialize.
        :param out: Pointer to the output base64 encoded string. "NULL" if empty or serialization fails.

    .. cpp:function:: void serialize(graph_hdf5* m_hdf5)

        Populates a graph_hdf5 struct with serialized data from this graph_t object.

        :param m_hdf5: Pointer to the graph_hdf5 struct to fill.

    .. cpp:function:: void meta_deserialize(std::map<std::string, int>* data, std::string* inpt)

        Deserializes a string representation back into a map (string to int).

        :param data: Pointer to the map to populate.
        :param inpt: Pointer to the input string (format: "key1|val1%key2|val2%..."). Handles "NULL".

    .. cpp:function:: void meta_deserialize(std::vector<torch::Tensor*>* data, std::string* inpt)

        Deserializes a base64 encoded string (created by meta_serialize) back into a vector of tensor pointers.

        :param data: Pointer to the vector of tensor pointers to populate. Allocates new tensors.
        :param inpt: Pointer to the input base64 encoded string. Handles "NULL".

    .. cpp:function:: torch::Tensor* meta_deserialize(std::string* inpt)

        Deserializes a base64 encoded string (created by meta_serialize) back into a single tensor pointer.

        :param inpt: Pointer to the input base64 encoded string. Handles "NULL".
        :return: Pointer to the newly allocated deserialized tensor, or nullptr if input is "NULL".

    .. cpp:function:: void deserialize(graph_hdf5* m_hdf5)

        Populates this graph_t object by deserializing data from a graph_hdf5 struct.

        :param m_hdf5: Pointer to the graph_hdf5 struct containing serialized data. Sets is_owner to true.

    .. cpp:function:: void _purge_data(std::vector<torch::Tensor*>* data)

        Deletes tensors pointed to by elements in a vector of tensor pointers.

        :param data: Pointer to the vector of tensor pointers. Clears the vector afterwards. Handles nullptr input.

    .. cpp:function:: void _purge_data(std::map<int, torch::Tensor*>* data)

        Deletes tensors pointed to by values in a map (int to tensor pointer).

        :param data: Pointer to the map. Clears the map afterwards. Handles nullptr input.

    .. cpp:function:: void _purge_data(std::map<int, std::vector<torch::Tensor*>*>* data)

        Deletes tensors pointed to within nested vectors managed by a map.

        :param data: Pointer to the map (int to vector of tensor pointers). Clears the map and vectors. Handles nullptr input.

    .. cpp:function:: std::vector<torch::Tensor*>* add_content(std::map<std::string, torch::Tensor*>* inpt)

        Converts a map of string-to-tensor pointers into a vector of tensor pointers, ordered by map iteration.

        :param inpt: Pointer to the input map (string -> tensor*).
        :return: Pointer to a newly allocated vector containing the tensor pointers from the map.

    .. cpp:function:: void _transfer_to_device( \
                                    std::vector<torch::Tensor>* trg, \
                                    std::vector<torch::Tensor*>* data, \
                                    torch::TensorOptions* dev \
                              )

        Helper function to transfer tensors from a source vector (CPU) to a target vector on a specified device.

        :param trg: Pointer to the target vector (on the specified device) to be populated.
        :param src: Pointer to the source vector of tensor pointers (on CPU).
        :param dev: Pointer to torch::TensorOptions specifying the target device.

        .. note:: Skips transfer if the target vector is already populated or the source is null.

    .. cpp:function:: torch::Tensor* return_any( \
                                    std::map<std::string, int>* loc, \
                                    std::map<int, std::vector<torch::Tensor>>* container, \
                                    std::string name, int dev_ \
                              )

        Helper function to retrieve a tensor from a device-specific container map.

        :param loc: Pointer to the name-to-index map for the feature type.
        :param container: Pointer to the map storing device-specific tensor vectors (device_idx -> vector<Tensor>).
        :param name: The name of the feature to retrieve.
        :param dev_: The index of the target device.
        :return: Pointer to the requested tensor on the specified device, or nullptr if not found.


.. cpp:function:: bool static fulltopo(particle_template*, particle_template*)

    Default topology function: connects every node to every other node (including self-loops).

    :param p1: Pointer to the first particle (unused).
    :param p2: Pointer to the second particle (unused).
    :return: Always returns true.


.. cpp:class:: graph_template : public tools

    Base class for defining how to construct a graph representation from event and particle data.

    This class provides the framework and tools to:
    1. Define nodes based on particles.
    2. Define the graph topology (edges).
    3. Add graph-level, node-level, and edge-level features (both truth and reconstructed).
    4. Implement event preselection logic.
    5. Compile the final graph data into a ``graph_t`` object.

    Users should inherit from this class and override virtual methods like ``CompileEvent`` and ``PreSelection``.

    .. cpp:function:: graph_template()

        Constructor: Initializes internal structures and properties.
        Sets up tensor options for CPU and configures property delegates.

    .. cpp:function:: virtual ~graph_template()

        Virtual destructor: Cleans up allocated resources (e.g., tensor options).

    .. cpp:function:: virtual graph_template* clone()

        Virtual clone method: Creates a new instance of the derived graph_template class.

        :return: Pointer to a new graph_template object (caller owns the memory).

        .. note:: Derived classes should override this to return an instance of their own type.

    .. cpp:function:: virtual void CompileEvent()

        Virtual method to compile graph features after nodes and basic topology are defined.

        This is the primary method users should override in derived classes.
        Inside this method, call ``add_graph_feature``, ``add_node_feature``, ``add_edge_feature``
        (and their truth/data variants) to populate the graph with relevant information.

    .. cpp:function:: virtual bool PreSelection()

        Virtual method to apply preselection criteria to the event.

        :return: True if the event passes preselection, false otherwise.

        .. note::
            Derived classes should override this to implement specific selection cuts.
            If it returns false, the event processing might be skipped.

    .. cpp:function:: void define_particle_nodes(std::vector<particle_template*>* prt)

        Defines the nodes of the graph based on a list of particles.
        Assigns a unique integer index to each unique particle based on its hash.
        Populates internal ``nodes`` and ``node_particles`` maps.

        :param prt: Pointer to a vector of particle_template pointers.

    .. cpp:function:: void define_topology(std::function<bool(particle_template*, particle_template*)> fx)

        Defines the graph topology (edges) based on a custom function.
        Iterates through all pairs of defined nodes and calls the provided function ``fx``.
        If ``fx(p1, p2)`` returns true, an edge is created between the nodes corresponding to p1 and p2.
        Populates internal topology representations (``_topology``, ``_topological_index``, ``m_topology``).

        :param fx: A function (or lambda) that takes two particle_template pointers and returns true if an edge should exist between them.

    .. cpp:function:: void flush_particles()

        Clears all particle-related information (nodes, topology) and resets the event pointer.
        Called internally to prepare the template for processing a new event.

    .. cpp:function:: bool operator == (graph_template& p)

        Compares two graph_template objects based on their event hash.

        :param p: The other graph_template object to compare against.
        :return: True if the event hashes are identical, false otherwise.

    .. cpp:member:: cproperty<long, graph_template> index
        Property for event index.

    .. cpp:member:: cproperty<double, graph_template> weight
        Property for event weight.

    .. cpp:member:: cproperty<bool, graph_template> preselection
        Property for preselection status (get/set).

    .. cpp:member:: cproperty<std::string, graph_template> hash
        Property for event hash (read-only).

    .. cpp:member:: cproperty<std::string, graph_template> tree
        Property for the name of the tree the event came from (read-only).

    .. cpp:member:: cproperty<std::string, graph_template> name
        Property for the name assigned to this graph type (get/set).

    .. cpp:member:: int threadIdx = -1
        Index of the processing thread assigned to this graph (if used in multithreading).

    .. cpp:member:: std::string filename = ""
        Name of the file the current event originates from.

    .. cpp:member:: meta* meta_data = nullptr
        Pointer to associated metadata object.

    .. cpp:function:: template <typename G> \
                         G* get_event()

        Gets a pointer to the underlying event_template object.

        :tparam G: The specific type of the event_template (or derived class).
        :return: Pointer to the event_template object, cast to type G.

    .. cpp:function:: template <typename G, typename O, typename X> \
                         void add_graph_truth_feature(O* ev, X fx, std::string name)

        Adds a truth-level graph feature derived from the event object.

        :tparam G: The data type of the feature (e.g., float, int, bool).
        :tparam O: The type of the event object (e.g., event_template or derived).
        :tparam X: The type of the getter function/lambda.
        :param ev: Pointer to the event object.
        :param fx: A getter function or lambda ``void(G*, O*)`` that retrieves the feature value.
        :param name: The name for this feature (e.g., "MET"). "T-" prefix is added automatically.

    .. cpp:function:: template <typename G, typename O, typename X> \
                         void add_graph_data_feature(O* ev, X fx, std::string name)

        Adds a data-level (reconstructed) graph feature derived from the event object.

        :tparam G: The data type of the feature (e.g., float, int, bool).
        :tparam O: The type of the event object (e.g., event_template or derived).
        :tparam X: The type of the getter function/lambda.
        :param ev: Pointer to the event object.
        :param fx: A getter function or lambda ``void(G*, O*)`` that retrieves the feature value.
        :param name: The name for this feature (e.g., "RecoMET"). "D-" prefix is added automatically.

    .. cpp:function:: template <typename G, typename O, typename X> \
                         void add_node_truth_feature(X fx, std::string name)

        Adds a truth-level node feature derived from each particle defined as a node.

        :tparam G: The data type of the feature (e.g., float, int, bool).
        :tparam O: The type of the particle object (e.g., particle_template or derived).
        :tparam X: The type of the getter function/lambda.
        :param fx: A getter function or lambda ``void(G*, O*)`` that retrieves the feature value from a particle.
        :param name: The name for this feature (e.g., "Charge"). "T-" prefix is added automatically.

    .. cpp:function:: template <typename G, typename O, typename X> \
                         void add_node_data_feature(X fx, std::string name)

        Adds a data-level (reconstructed) node feature derived from each particle defined as a node.

        :tparam G: The data type of the feature (e.g., float, int, bool).
        :tparam O: The type of the particle object (e.g., particle_template or derived).
        :tparam X: The type of the getter function/lambda.
        :param fx: A getter function or lambda ``void(G*, O*)`` that retrieves the feature value from a particle.
        :param name: The name for this feature (e.g., "RecoCharge"). "D-" prefix is added automatically.

    .. cpp:function:: template <typename G, typename O, typename X> \
                         void add_edge_truth_feature(X fx, std::string name)

        Adds a truth-level edge feature derived from pairs of connected particles.
        Requires ``define_topology`` to have been called first. If not, uses full topology.

        :tparam G: The data type of the feature (e.g., float, int, bool).
        :tparam O: The type of the particle object (e.g., particle_template or derived).
        :tparam X: The type of the getter function/lambda.
        :param fx: A getter function or lambda ``void(G*, std::tuple<O*, O*>*)`` that retrieves the feature value from a pair of particles.
        :param name: The name for this feature (e.g., "DeltaR"). "T-" prefix is added automatically.

    .. cpp:function:: template <typename G, typename O, typename X> \
                         void add_edge_data_feature(X fx, std::string name)

        Adds a data-level (reconstructed) edge feature derived from pairs of connected particles.
        Requires ``define_topology`` to have been called first. If not, uses full topology.

        :tparam G: The data type of the feature (e.g., float, int, bool).
        :tparam O: The type of the particle object (e.g., particle_template or derived).
        :tparam X: The type of the getter function/lambda.
        :param fx: A getter function or lambda ``void(G*, std::tuple<O*, O*>*)`` that retrieves the feature value from a pair of particles.
        :param name: The name for this feature (e.g., "RecoDeltaR"). "D-" prefix is added automatically.

    .. cpp:function:: bool double_neutrino( \
                                    double mass_top = 172.62*1000, double mass_wboson = 80.385*1000, \
                                    double top_perc = 0.85, double w_perc = 0.95, double distance = 1e-8, int steps = 10 \
                              )

        Attempts to solve the double neutrino ambiguity for ttbar events using a combinatorial approach.
        Requires specific node features ("D-pt", "D-eta", "D-phi", "D-energy", "D-is_lep", "D-is_b")
        and graph features ("D-met", "D-phi") to be defined beforehand.
        Uses the ``pyc::nusol::combinatorial`` function internally.

        :param mass_top: Assumed top quark mass (in MeV).
        :param mass_wboson: Assumed W boson mass (in MeV).
        :param top_perc: Percentage tolerance for top mass constraint.
        :param w_perc: Percentage tolerance for W boson mass constraint.
        :param distance: Minimum distance parameter for the solver.
        :param steps: Maximum number of steps for the solver.
        :return: True if the calculation was attempted (regardless of success), false if required features are missing.

        .. note:: The implementation currently calculates neutrino solutions but doesn't apply them back to the particles.

    // --- Private Members ---
    // Note: Sphinx C++ domain doesn't have explicit private markers like Doxygen.
    // Access control is implied by C++ rules. Friends are noted here.

    .. note::
        ``container`` and ``analysis`` are friend classes.

    // -------- Private Feature Adding Overloads --------
    // These are called by the public template methods (add_graph_truth_feature, etc.)
    // They handle the conversion of single values or vectors to torch::Tensor.

    .. cpp:function:: void add_graph_feature(bool, std::string)
        Adds a single boolean graph feature.
    .. cpp:function:: void add_graph_feature(std::vector<bool>, std::string)
        Adds a vector of boolean graph features.
    .. cpp:function:: void add_graph_feature(float, std::string)
        Adds a single float graph feature.
    .. cpp:function:: void add_graph_feature(std::vector<float>, std::string)
        Adds a vector of float graph features.
    .. cpp:function:: void add_graph_feature(double, std::string)
        Adds a single double graph feature.
    .. cpp:function:: void add_graph_feature(std::vector<double>, std::string)
        Adds a vector of double graph features.
    .. cpp:function:: void add_graph_feature(long, std::string)
        Adds a single long graph feature.
    .. cpp:function:: void add_graph_feature(std::vector<long>, std::string)
        Adds a vector of long graph features.
    .. cpp:function:: void add_graph_feature(int, std::string)
        Adds a single int graph feature.
    .. cpp:function:: void add_graph_feature(std::vector<int>, std::string)
        Adds a vector of int graph features.
    .. cpp:function:: void add_graph_feature(std::vector<std::vector<int>>, std::string)
        Adds a vector of vector of int graph features (e.g., for ragged tensors).

    .. cpp:function:: void add_node_feature(bool, std::string)
        Adds a single boolean node feature (broadcasted to all nodes).
    .. cpp:function:: void add_node_feature(std::vector<bool>, std::string)
        Adds a vector of boolean node features (one per node).
    .. cpp:function:: void add_node_feature(float, std::string)
        Adds a single float node feature (broadcasted to all nodes).
    .. cpp:function:: void add_node_feature(std::vector<float>, std::string)
        Adds a vector of float node features (one per node).
    .. cpp:function:: void add_node_feature(double, std::string)
        Adds a single double node feature (broadcasted to all nodes).
    .. cpp:function:: void add_node_feature(std::vector<double>, std::string)
        Adds a vector of double node features (one per node).
    .. cpp:function:: void add_node_feature(long, std::string)
        Adds a single long node feature (broadcasted to all nodes).
    .. cpp:function:: void add_node_feature(std::vector<long>, std::string)
        Adds a vector of long node features (one per node).
    .. cpp:function:: void add_node_feature(int, std::string)
        Adds a single int node feature (broadcasted to all nodes).
    .. cpp:function:: void add_node_feature(std::vector<int>, std::string)
        Adds a vector of int node features (one per node).
    .. cpp:function:: void add_node_feature(std::vector<std::vector<int>>, std::string)
        Adds a vector of vector of int node features (e.g., for ragged node features).

    .. cpp:function:: void add_edge_feature(bool, std::string)
        Adds a single boolean edge feature (broadcasted to all edges).
    .. cpp:function:: void add_edge_feature(std::vector<bool>, std::string)
        Adds a vector of boolean edge features (one per edge).
    .. cpp:function:: void add_edge_feature(float, std::string)
        Adds a single float edge feature (broadcasted to all edges).
    .. cpp:function:: void add_edge_feature(std::vector<float>, std::string)
        Adds a vector of float edge features (one per edge).
    .. cpp:function:: void add_edge_feature(double, std::string)
        Adds a single double edge feature (broadcasted to all edges).
    .. cpp:function:: void add_edge_feature(std::vector<double>, std::string)
        Adds a vector of double edge features (one per edge).
    .. cpp:function:: void add_edge_feature(long, std::string)
        Adds a single long edge feature (broadcasted to all edges).
    .. cpp:function:: void add_edge_feature(std::vector<long>, std::string)
        Adds a vector of long edge features (one per edge).
    .. cpp:function:: void add_edge_feature(int, std::string)
        Adds a single int edge feature (broadcasted to all edges).
    .. cpp:function:: void add_edge_feature(std::vector<int>, std::string)
        Adds a vector of int edge features (one per edge).
    .. cpp:function:: void add_edge_feature(std::vector<std::vector<int>>, std::string)
        Adds a vector of vector of int edge features (e.g., for ragged edge features).

    .. cpp:function:: template <typename G, typename g> \
                         torch::Tensor to_tensor(std::vector<G> _data, at::ScalarType _op, g prim)

        Converts a std::vector of a primitive type to a torch::Tensor.

        :tparam G: The primitive data type in the vector (e.g., float, int).
        :tparam g: The primitive data type again (used for template deduction).
        :param _data: The input vector.
        :param _op: The desired torch::ScalarType (e.g., torch::kFloat).
        :param prim: A dummy primitive value of type g (used for type deduction).
        :return: A torch::Tensor containing the data from the vector.

    // --- Static Property Setters/Getters ---
    // Used internally by the cproperty delegates.
    .. cpp:function:: void static set_name(std::string*, graph_template*)
        Static setter for the graph name property.
    .. cpp:function:: void static set_preselection(bool*, graph_template*)
        Static setter for the preselection property.
    .. cpp:function:: void static get_hash(std::string*, graph_template*)
        Static getter for the event hash property.
    .. cpp:function:: void static get_index(long*, graph_template*)
        Static getter for the event index property.
    .. cpp:function:: void static get_weight(double*, graph_template*)
        Static getter for the event weight property.
    .. cpp:function:: void static get_tree(std::string*, graph_template*)
        Static getter for the tree name property.
    .. cpp:function:: void static get_preselection(bool*, graph_template*)
        Static getter for the preselection property.

    .. cpp:function:: void static build_export( \
                                    std::map<std::string, torch::Tensor*>* _truth_t, std::map<std::string, int>* _truth_i, \
                                    std::map<std::string, torch::Tensor*>* _data_t , std::map<std::string, int>*  _data_i, \
                                    std::map<std::string, torch::Tensor>* _fx \
                              )

        Static helper function to populate export maps for graph_t creation.
        Iterates through a feature map (_fx) and populates the corresponding
        truth or data maps (_truth_t, _truth_i or _data_t, _data_i) based on the feature name prefix ("T-" or "D-").

        :param _truth_t: Pointer to the map storing truth tensor pointers.
        :param _truth_i: Pointer to the map storing truth feature name-to-index mapping.
        :param _data_t: Pointer to the map storing data tensor pointers.
        :param _data_i: Pointer to the map storing data feature name-to-index mapping.
        :param _fx: Pointer to the source feature map (graph_fx, node_fx, or edge_fx).

    // --- Internal Data Members ---
    .. cpp:member:: int num_nodes = 0
        Number of nodes defined for the current graph.
    .. cpp:member:: std::map<std::string, int> nodes = {}
        Maps particle hash strings to unique node indices.
    .. cpp:member:: std::map<int, particle_template*> node_particles = {}
        Maps unique node indices back to particle pointers.
    .. cpp:member:: std::map<std::string, torch::Tensor> graph_fx = {}
        Stores graph-level features as tensors.
    .. cpp:member:: std::map<std::string, torch::Tensor> node_fx  = {}
        Stores node-level features as tensors.
    .. cpp:member:: std::map<std::string, torch::Tensor> edge_fx  = {}
        Stores edge-level features as tensors.

    .. cpp:member:: std::vector<std::vector<int>> _topology
        Adjacency list representation of the topology (stores pairs {src, dst}).
    .. cpp:member:: std::vector<int> _topological_index
        Maps the dense pair index (itr1*N + itr2) to the sparse edge index, or -1 if no edge exists.
    .. cpp:member:: torch::Tensor m_topology
        Topology stored as a [2, num_edges] tensor (edge_index format).

    .. cpp:member:: torch::TensorOptions* op = nullptr
        Tensor options, typically configured for CPU.
    .. cpp:member:: event_template* m_event = nullptr
        Pointer to the current event being processed.

    .. cpp:member:: bool m_preselection = true
        Internal storage for the preselection status.

    .. cpp:function:: graph_template* build(event_template* el)

        Internal build method called by analysis or container to create a graph instance for an event.
        Clones the template, associates it with the event, and copies basic event data.

        :param ev: Pointer to the event_template object.
        :return: Pointer to the newly created graph_template instance configured for the event.

    .. cpp:function:: graph_t* data_export()

        Exports the compiled graph data into a graph_t structure.
        Calls ``build_export`` to organize features and creates a new graph_t object.

        :return: Pointer to a newly allocated graph_t object containing the graph data (caller owns the memory).

    .. cpp:member:: event_t data
        Local copy of the basic event data (hash, index, weight, etc.).

