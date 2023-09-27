.. _data-types:

Data Types and Dictionary Mapping
*********************************

Introduction
____________
Since most of the backend of Analysis-G is built in C++ and interfaced via Cython, multiple data types were used to simplify passing arguments between layers.
Luckily, Cython and C++ can share py:type:struct types, which can be mapped into dictionaries and back.
This section of the documentation outlines these types with their respective key and value pairs.

.. py:data:: code_t

    :var input_params: list[str]
    :var co_vars: list[str]
    :var param_space: dict[str, str]
    :var trace: dict[str, list[str]]
    :var extern_imports: dict[str, list[str]]
    :var dependency_hashes: list[str]
    :var function_name: str
    :var class_name: str
    :var hash: str
    :var source_code: str
    :var object_code: str
    :var defaults: str
    :var is_class: bool
    :var is_function: bool
    :var is_callable: bool
    :var is_initialized: bool
    :var has_param_variable: bool

.. py:data:: leaf_t

    :var requested: str
    :var matched: str
    :var branch_name: str
    :var tree_name: str
    :var path: str

.. py:data:: branch_t

    :var requested: str
    :var matched: str
    :var tree_name: str
    :var leaves: list[leaf_t]

.. py:data:: tree_t

    :var size: int
    :var requested: str
    :var matched: str
    :var branches: list[branch_t]
    :var leaves: list[leaf_t]

.. py:data:: meta_t

    :var hash: str
    :var original_input: str
    :var original_path: str
    :var original_name: str

    :var req_trees: list[str]
    :var req_branches: list[str]
    :var req_leaves: list[str]
    :var mis_trees: list[str]
    :var mis_branches: list[str]
    :var mis_leaves: list[str]

    :var dsid: int
    :var AMITag: str
    :var generators: str
    :var isMC: bool
    :var derivationFormat: str

    :var inputrange: dict[int, int]
    :var inputfiles: dict[int, str]
    :var config: dict[str, str]

    :var eventNumber: int
    :var event_index: int
    :var found: bool
    :var DatasetName: str
    :var ecmEnergy: float
    :var genFiltEff: float
    :var completion: float
    :var beam_energy: float
    :var crossSection: float
    :var crossSection_mean: float
    :var totalSize: float
    :var nFiles: int
    :var run_number: int
    :var totalEvents: int
    :var datasetNumber: int
    :var identifier: str
    :var prodsysStatus: str
    :var dataType: str
    :var version: str
    :var PDF: str
    :var AtlasRelease: str
    :var principalPhysicsGroup: str
    :var physicsShort: str
    :var generatorName: str
    :var geometryVersion: str
    :var conditionsTag: str
    :var generatorTune: str
    :var amiStatus: str
    :var beamType: str
    :var productionStep: str
    :var projectName: str
    :var statsAlgorithm: str
    :var genFilterNames: str
    :var file_type: str
    :var sample_name: str
    :var keywords: list[str]
    :var weights: list[str]
    :var keyword: list[str]
    :var LFN: dict[str, int]
    :var fileGUID: list[str]
    :var events: list[int]
    :var fileSize: list[float]


.. py:data:: particle_t

    :var e: float
    :var mass: float
    :var px: float
    :var py: float
    :var pz: float
    :var pt: float
    :var eta: float
    :var phi: float
    :var cartesian: bool
    :var polar: bool
    :var charge: float
    :var pdgid: int
    :var index: int
    :var type: str
    :var hash: str
    :var symbol: str
    :var lepdef: list[int]
    :var nudef: list[int]

.. py:data:: event_t

    :var event_name: str
    :var commit_hash: str
    :var code_hash: str
    :var deprecated: bool
    :var cached: bool
    :var weight: float
    :var event_index: int
    :var event_hash: str
    :var event_tagging: str
    :var event_tree: str
    :var event_root: str
    :var pickled_data: str
    :var graph: bool
    :var selection: bool
    :var event: bool

.. py:data:: graph_t

    :var event_name: str
    :var code_hash: str
    :var errors: dict[str, str]
    :var presel: dict[str, int]
    :var cached: bool
    :var event_index: int
    :var weight: float
    :var event_hash: str
    :var event_tagging: str
    :var event_tree: str
    :var event_root: str
    :var pickled_data: str
    :var train: bool
    :var evaluation: bool
    :var validation: bool
    :var empty_graph: bool
    :var skip_graph: bool
    :var src_dst: dict[str, list[int]]
    :var hash_particle: dict[str, int]
    :var self_loops: bool
    :var graph_feature: dict[str, str]
    :var node_feature: dict[str, str]
    :var edge_feature: dict[str, str]
    :var pre_sel_feature: dict[str, str]
    :var topo_hash: str
    :var graph: bool
    :var selection: bool
    :var event: bool

.. py:data:: selection_t

    :var event_name: str
    :var code_hash: str
    :var errors: dict[str, int]
    :var cached: bool
    :var event_index: int
    :var weight: float
    :var event_hash: str
    :var event_tagging: str
    :var event_tree: str
    :var event_root: str
    :var pickled_data: str
    :var pickled_strategy_data: str
    :var strat_merge: dict[str, str]
    :var data_merge: dict[str, str]
    :var cutflow: dict[str, int]
    :var timestats: list[float]
    :var all_weights: list[float]
    :var selection_weights: list[float]
    :var allow_failure: bool
    :var _params_: str
    :var graph: bool
    :var selection: bool
    :var event: bool

.. py:data:: batch_t

    :var events: dict[str, event_t]
    :var graphs: dict[str, graphs_t]
    :var selections: dict[str, selection_t]
    :var code_hashes: dict[str, code_t]
    :var meta: meta_t
    :var hash: str

.. py:data:: folds_t

    :var test: bool
    :var train: bool
    :var evaluation: bool
    :var kfold: int
    :var event_hash: str

.. py:data:: data_t

    :var name: str
    :var truth: list[list[float]]
    :var pred: list[list[float]]
    :var index: list[list[float]]
    :var nodes: list[list[float]]
    :var loss: list[list[float]]
    :var accuracy: list[list[float]]

.. py:data:: metric_t
   
    :var truth: dict[str, list[list[float]]
    :var pred: dict[str, list[list[float]]
    :var acc_average: dict[str, float]
    :var loss_average: dict[str, float]
    :var num_nodes: dict[str, int]


.. py:data:: root_t

    :var batches: dict[str, batch_t]
    :var n_events: dict[str, int]
    :var n_graphs: dict[str, int]
    :var n_selections: dict[str, int]

.. py:data:: tracer_t

    :var root_names: dict[str, root_t]
    :var root_meta: dict[str, meta_t]
    :var hashed_code: dict[str, code_t]
    :var event_trees: dict[str, int]
    :var link_event_code: dict[str, str]
    :var link_graph_code: dict[str, str]

.. py:data:: export_t

    :var root_meta: dict[str, meta_t]
    :var hashed_code: dict[str, code_t]
    :var link_event_code: dict[str, str]
    :var link_graph_code: dict[str, str]
    :var link_selection_code: dict[str, str]
    :var event_name_hash: dict[str, dict[str]]
    :var graph_name_hash: dict[str, dict[str]]
    :var selection_name_hash: dict[str, dict[str]]
    :var event_dir: dict[str, str]
    :var graph_dir: dict[str, str]
    :var selection_dir: dict[str, str]

.. py:data:: settings_t

    :var projectname: str
    :var outputdirectory: str
    :var files: dict[str, list[str]]
    :var samplemap: dict[str, list[str]]
    :var verbose: int
    :var chunks: int
    :var threads: int
    :var enable_pyami: bool
    :var tree: str
    :var eventname: str
    :var graphname: str
    :var selectionname: str
    :var event_start: int
    :var event_stop: int
    :var training_name: str
    :var run_name: str
    :var device: str
    :var optimizer_name: str
    :var optimizer_params: dict[str, str]
    :var scheduler_name: str
    :var scheduler_params: dict[str, str]
    :var kfolds: int
    :var batch_size: int
    :var epochs: int
    :var epoch: dict[int, int]
    :var kfold: list[int]
    :var model: code_t
    :var model_params: dict[str, str]
    :var kinematic_map: dict[str, str]
    :var debug_mode: bool
    :var continue_training: bool
    :var runplotting: bool
    :var sort_by_nodes: bool
    :var enable_reconstruction: bool
    :var getgraph: bool
    :var getevent: bool
    :var getselection: bool
    :var eventcache: bool
    :var graphcache: bool
    :var search: list[str]
    :var get_all: bool
    :var hashed_code: dict[str, code_t]
    :var link_event_code: dict[str, str]
    :var link_graph_code: dict[str, str]
    :var link_selection_code: dict[str, str]

