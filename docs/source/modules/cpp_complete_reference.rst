C++ Modules Complete Reference
===============================

This document provides comprehensive documentation for all 23 C++ modules in AnalysisG.

**Location**: ``src/AnalysisG/modules/*/``

Module Overview
---------------

The modules are organized into categories:

1. **Core Templates**: particle, event, graph, selection, metric, model, lossfx
2. **Data Management**: io, container, dataloader, sampletracer  
3. **Analysis**: analysis, optimizer, metrics, plotting, roc
4. **Physics**: nusol (neutrino reconstruction)
5. **Infrastructure**: structs, tools, typecasting, notification, meta

Each module is a self-contained C++ library with headers in ``include/`` and implementations in ``cxx/``.

Core Template Modules
---------------------

particle
~~~~~~~~

**Header**: ``modules/particle/include/templates/particle_template.h``

Four-momentum particle representation with kinematics and identification.

**Key Class**: ``particle_template``

.. code-block:: cpp

   class particle_template : public tools {
   public:
       // Constructors
       particle_template();
       particle_template(particle_t* p);
       particle_template(double px, double py, double pz, double e);
       particle_template(double px, double py, double pz);  // massless
       
       // Four-momentum properties (cproperty provides getters/setters)
       cproperty<double, particle_template> px, py, pz, e;
       cproperty<double, particle_template> pt, eta, phi, mass;
       cproperty<double, particle_template> P;     // momentum magnitude
       cproperty<double, particle_template> beta;  // velocity β = P/E
       
       // Identification
       cproperty<int, particle_template> pdgid;
       cproperty<double, particle_template> charge;
       cproperty<std::string, particle_template> symbol;  // "e-", "μ+", etc.
       cproperty<std::string, particle_template> type;    // "jet", "electron", etc.
       cproperty<std::string, particle_template> hash;    // unique identifier
       cproperty<int, particle_template> index;
       
       // Boolean type flags
       cproperty<bool, particle_template> is_b;         // b-quark/jet
       cproperty<bool, particle_template> is_lep;       // lepton
       cproperty<bool, particle_template> is_nu;        // neutrino
       cproperty<bool, particle_template> is_add;       // additional particle
       cproperty<bool, particle_template> lep_decay;    // from leptonic decay
       
       // Decay chain
       cproperty<std::map<std::string, particle_template*>, particle_template> parents;
       cproperty<std::map<std::string, particle_template*>, particle_template> children;
       
       // Methods
       double DeltaR(particle_template* p);  // Angular separation
       bool is(std::vector<int> pdgids);      // Check if PDG ID in list
       void to_cartesian();                    // Convert from polar
       void to_polar();                        // Convert to polar
       
       // Operators
       bool operator==(particle_template& p);
       template<typename g> g operator+(g& p);  // Four-momentum addition
       void operator+=(particle_template* p);
       void iadd(particle_template* p);
       
       // Serialization
       std::map<std::string, std::map<std::string, particle_t>> __reduce__();
       
       // Decay chain management
       bool register_parent(particle_template* p);
       bool register_child(particle_template* p);
       std::map<std::string, particle_template*> m_parents;
       std::map<std::string, particle_template*> m_children;
       
       // ROOT variable mapping
       void add_leaf(std::string key, std::string leaf = "");
       std::map<std::string, std::string> leaves;
       
       // Cloning and building
       virtual particle_template* clone();
       virtual void build(std::map<std::string, particle_template*>* event, element_t* el);
       
       particle_t data;  // Underlying data structure
   };

**Usage**:

.. code-block:: cpp

   particle_template* electron = new particle_template();
   electron->px = 25000.0;  // MeV
   electron->py = 15000.0;
   electron->pz = 40000.0;
   electron->e  = 50000.0;
   
   double pt_val = electron->pt;   // Computed: √(px²+py²)
   double eta_val = electron->eta; // Computed: -ln(tan(θ/2))
   
   electron->pdgid = 11;     // Electron PDG ID
   electron->charge = -1.0;
   electron->type = "electron";
   
   // Check if it's a lepton
   bool is_lepton = electron->is_lep;  // true for e, μ, τ
   
   // Compute angular separation
   double dr = electron->DeltaR(muon);

event
~~~~~

**Header**: ``modules/event/include/templates/event_template.h``

Event-level container managing particles and metadata.

**Key Class**: ``event_template``

.. code-block:: cpp

   class event_template : public tools {
   public:
       event_template();
       virtual ~event_template();
       
       // ROOT tree configuration
       cproperty<std::vector<std::string>, event_template> trees;
       cproperty<std::vector<std::string>, event_template> branches;
       cproperty<std::vector<std::string>, event_template> leaves;
       cproperty<std::string, event_template> tree;
       
       // Event identification
       cproperty<std::string, event_template> name;
       cproperty<std::string, event_template> hash;
       cproperty<double, event_template> weight;
       cproperty<long, event_template> index;
       
       // Particle registration
       template<typename G>
       void register_particle(std::map<std::string, G*>* object);
       
       template<typename G>
       void deregister_particle(std::map<std::string, G*>* object);
       
       // Virtual methods for inheritance
       virtual event_template* clone();
       virtual void build(element_t* el);
       virtual void CompileEvent();
       
       // Event building
       std::map<std::string, event_template*> build_event(
           std::map<std::string, data_t*>* evnt
       );
       
       // Neutrino reconstruction
       std::vector<particle_template*> double_neutrino(
           std::vector<particle_template*>* targets,
           double phi, double met, double limit = 1e3
       );
       
       // Operators
       bool operator==(event_template& p);
       
       // Data members
       event_t data;
       meta* meta_data;
       std::string filename;
       
       void add_leaf(std::string key, std::string leaf = "");
       void flush_particles();
       
   private:
       void build_mapping(std::map<std::string, data_t*>* evnt);
       void flush_leaf_string();
       
       std::map<std::string, bool> next_;
       std::map<std::string, particle_template*> particle_generators;
       std::map<std::string, std::map<std::string, element_t>> tree_variable_link;
       std::map<std::string, std::map<std::string, particle_template*>*> particle_link;
       std::map<std::string, particle_template*> garbage;
   };

graph
~~~~~

**Header**: ``modules/graph/include/templates/graph_template.h``

Graph representation for GNN processing.

**Key Class**: ``graph_template``

.. code-block:: cpp

   class graph_template : public tools {
   public:
       graph_template();
       virtual ~graph_template();
       
       // Graph structure
       cproperty<std::string, graph_template> name;
       cproperty<std::string, graph_template> hash;
       
       // Virtual methods
       virtual graph_template* clone();
       virtual void build(event_template* ev);
       
       // Graph data
       graph_t data;
       
       // Node and edge management
       void add_node(particle_template* p);
       void add_edge(int src, int dst, double weight = 1.0);
       
       // Adjacency operations
       void build_adjacency();
       std::vector<std::vector<int>> get_adjacency_list();
       
       bool operator==(graph_template& g);
   };

selection
~~~~~~~~~

**Header**: ``modules/selection/include/templates/selection_template.h``

Custom selection/filtering logic.

**Key Class**: ``selection_template``

.. code-block:: cpp

   class selection_template : public tools {
   public:
       selection_template();
       virtual ~selection_template();
       
       cproperty<std::string, selection_template> name;
       cproperty<std::string, selection_template> hash;
       
       virtual selection_template* clone();
       virtual void selection(event_template* ev);
       virtual void selection(graph_template* gr);
       
       selection_t data;
       bool operator==(selection_template& s);
   };

metric
~~~~~~

**Header**: ``modules/metric/include/templates/metric_template.h``

Evaluation metrics for model performance.

**Key Class**: ``metric_template``

.. code-block:: cpp

   class metric_template : public tools {
   public:
       metric_template();
       virtual ~metric_template();
       
       cproperty<std::string, metric_template> name;
       
       virtual metric_template* clone();
       virtual void calculate(/* model outputs */);
       
       metric_t data;
   };

model
~~~~~

**Header**: ``modules/model/include/templates/model_template.h``

Machine learning model template.

**Key Class**: ``model_template``

.. code-block:: cpp

   class model_template : public tools {
   public:
       model_template();
       virtual ~model_template();
       
       cproperty<std::string, model_template> name;
       
       virtual model_template* clone();
       virtual void forward(/* inputs */);
       virtual void backward(/* gradients */);
       
       model_t data;
       
       // State management
       void save_state(std::string path);
       void load_state(std::string path);
   };

lossfx
~~~~~~

**Header**: ``modules/lossfx/include/templates/lossfx.h``

Loss function definitions.

**Key Class**: ``lossfx``

.. code-block:: cpp

   class lossfx : public tools {
   public:
       lossfx();
       virtual ~lossfx();
       
       cproperty<std::string, lossfx> name;
       
       virtual double calculate(/* predictions, targets */);
       virtual void backward(/* gradients */);
       
       loss_t data;
   };

Data Management Modules
-----------------------

io
~~

**Header**: ``modules/io/include/io/io.h``

Input/output operations for ROOT and HDF5 files.

**Key Class**: ``io``

.. code-block:: cpp

   class io : public tools, public notification {
   public:
       io();
       ~io();
       
       // HDF5 operations
       template<typename g>
       void write(std::vector<g>* inpt, std::string set_name);
       
       template<typename g>
       void write(g* inpt, std::string set_name);
       
       template<typename g>
       void read(std::vector<g>* outpt, std::string set_name);
       
       template<typename g>
       void read(g* out, std::string set_name);
       
       // File management
       bool start(std::string filename, std::string read_write);
       void end();
       std::vector<std::string> dataset_names();
       
       // ROOT operations
       std::map<std::string, long> root_size();
       void check_root_file_paths();
       bool scan_keys();
       void root_begin();
       void root_end();
       void trigger_pcm();
       void import_settings(settings_t* params);
       std::map<std::string, data_t*>* get_data();
       
       // Configuration
       bool enable_pyami;
       std::string metacache_path;
       std::string current_working_path;
       // ... (many more members)
   };

**Features**:

* Reads ROOT TTrees with branch/leaf access
* Writes/reads HDF5 datasets
* Automatic type detection
* Metadata caching
* Progress tracking

container
~~~~~~~~~

**Header**: ``modules/container/include/container/container.h``

Container for managing event/graph/selection collections.

**Key Class**: ``container``

.. code-block:: cpp

   class container : public tools {
   public:
       container();
       ~container();
       
       // Metadata
       void add_meta_data(meta*, std::string);
       meta* get_meta_data();
       
       // Template management
       bool add_selection_template(selection_template*);
       bool add_event_template(event_template*, std::string label);
       bool add_graph_template(graph_template*, std::string label);
       
       // Population
       void fill_selections(std::map<std::string, selection_template*>* inpt);
       void get_events(std::vector<event_template*>*, std::string label);
       void populate_dataloader(dataloader* dl);
       
       // Processing
       void compile(size_t* len, int threadIdx);
       size_t len();
       entry_t* add_entry(std::string hash);
       
       // Data members
       meta* meta_data;
       std::string* filename;
       std::string* output_path;
       std::string label;
       std::map<std::string, entry_t> random_access;
       std::map<std::string, selection_template*>* merged;
   };

**Helper Struct**: ``entry_t``

.. code-block:: cpp

   struct entry_t {
       std::string hash;
       std::vector<graph_t*> m_data;
       std::vector<graph_template*> m_graph;
       std::vector<event_template*> m_event;
       std::vector<selection_template*> m_selection;
       
       void init();
       void destroy();
       bool has_event(event_template* ev);
       bool has_graph(graph_template* gr);
       bool has_selection(selection_template* sel);
   };

dataloader
~~~~~~~~~~

**Header**: ``modules/dataloader/include/generators/dataloader.h``

Batch data loading for training.

**Key Class**: ``dataloader``

.. code-block:: cpp

   class dataloader : public tools {
   public:
       dataloader();
       ~dataloader();
       
       // Batch operations
       void add_batch(/* data */);
       void shuffle();
       void reset();
       
       // Configuration
       void set_batch_size(size_t size);
       void set_num_workers(int workers);
       
       // Iteration
       bool next_batch(/* output */);
       size_t num_batches();
       
       dataloader_t data;
   };

sampletracer
~~~~~~~~~~~~

**Header**: ``modules/sampletracer/include/generators/sampletracer.h``

Tracks sample metadata (cross-sections, weights).

**Key Class**: ``sampletracer``

.. code-block:: cpp

   class sampletracer : public tools {
   public:
       sampletracer();
       ~sampletracer();
       
       // Sample tracking
       void add_sample(std::string label, double xsec, double weight);
       double get_cross_section(std::string label);
       double get_sum_of_weights(std::string label);
       
       // Normalization
       double get_scale_factor(std::string label, double lumi);
       
       sampletracer_t data;
   };

Analysis Modules
----------------

analysis
~~~~~~~~

**Header**: ``modules/analysis/include/AnalysisG/analysis.h``

Main orchestrator coordinating all analysis components.

**Key Class**: ``analysis``

.. code-block:: cpp

   class analysis : public notification, public tools {
   public:
       analysis();
       ~analysis();
       
       // Configuration
       void add_samples(std::string path, std::string label);
       void add_selection_template(selection_template* sel);
       void add_event_template(event_template* ev, std::string label);
       void add_graph_template(graph_template* gr, std::string label);
       void add_metric_template(metric_template* mx, model_template* mdl);
       void add_model(model_template* model, optimizer_params_t* op, 
                      std::string run_name);
       void add_model(model_template* model, std::string run_name);  // inference
       
       // Execution
       void start();
       void attach_threads();
       
       // Progress tracking
       std::map<std::string, std::vector<float>> progress();
       std::map<std::string, std::string> progress_mode();
       std::map<std::string, std::string> progress_report();
       std::map<std::string, bool> is_complete();
       
       // Configuration
       settings_t m_settings;
       std::map<std::string, meta*> meta_data;
       
   private:
       void check_cache();
       void build_project();
       void build_events();
       void build_selections();
       void build_graphs();
       void build_model_session();
       void build_inference();
       bool build_metric();
       void build_metric_folds();
       void build_dataloader(bool training);
       void fetchtags();
       
       // Threading
       static void execution(/* params */);
       static void execution_metric(/* params */);
       static void initialize_loop(/* params */);
       
       // Internal state
       std::map<std::string, std::string> file_labels;
       std::map<std::string, event_template*> event_labels;
       std::map<std::string, metric_template*> metric_names;
       std::map<std::string, selection_template*> selection_names;
       std::map<std::string, std::map<std::string, graph_template*>> graph_labels;
       
       std::vector<std::string> model_session_names;
       std::map<std::string, model_template*> model_inference;
       std::map<std::string, model_template*> model_metrics;
       std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions;
       
       std::map<std::string, optimizer*> trainer;
       std::map<std::string, model_report*> reports;
       std::vector<std::thread*> threads;
       
       std::vector<folds_t>* tags;
       dataloader* loader;
       sampletracer* tracer;
       io* reader;
       
       bool started;
   };

optimizer
~~~~~~~~~

**Header**: ``modules/optimizer/include/generators/optimizer.h``

Training optimization (Adam, SGD, etc.).

**Key Class**: ``optimizer``

.. code-block:: cpp

   class optimizer : public tools {
   public:
       optimizer();
       ~optimizer();
       
       // Configuration
       void set_learning_rate(double lr);
       void set_momentum(double momentum);
       void set_weight_decay(double decay);
       
       // Training
       void zero_grad();
       void step();
       
       // State
       void save_state(std::string path);
       void load_state(std::string path);
       
       optimizer_t data;
       optimizer_params_t params;
   };

metrics
~~~~~~~

**Header**: ``modules/metrics/include/metrics/metrics.h``

Metric computation utilities.

plotting
~~~~~~~~

**Header**: ``modules/plotting/include/plotting/plotting.h``

Plotting utilities for analysis results.

roc
~~~

**Header**: ``modules/roc/include/plotting/roc.h``

ROC curve computation and plotting.

Physics Module
--------------

nusol
~~~~~

**Location**: ``modules/nusol/nusol/include/reconstruction/nusol.h``

Neutrino momentum reconstruction algorithms.

**Key Features**:

* Analytical W→lν solver
* Numerical ttbar→lνblνb solver
* Multiple solution handling
* Constraint optimization

Infrastructure Modules
----------------------

structs
~~~~~~~

**Location**: ``modules/structs/include/structs/``

Core data structures used throughout.

**Key Structures**:

.. code-block:: cpp

   struct particle_t {
       double px, py, pz, e;
       int pdgid, index;
       double charge;
       std::string type;
       bool polar;
       // ... more fields
   };
   
   struct event_t {
       long index;
       double weight;
       std::string hash, name, tree;
       std::vector<std::string> trees, branches;
       // ... more fields
   };
   
   struct graph_t {
       std::vector<long> edge_index;
       std::vector<double> edge_attr;
       std::vector<double> node_attr;
       // ... more fields
   };
   
   struct settings_t {
       std::string output_path;
       std::string sow_name;
       int kfolds;
       std::vector<int> kfold;
       // ... more fields
   };

tools
~~~~~

**Header**: ``modules/tools/include/tools/tools.h``

Utility functions and base class.

**Key Class**: ``tools``

.. code-block:: cpp

   class tools {
   public:
       tools();
       virtual ~tools();
       
       // String operations
       std::string env(const std::string& s);
       std::string enc(const std::string& s);
       
       // Vector operations
       template<typename T>
       std::vector<T> flatten(std::vector<std::vector<T>> v);
       
       // Hashing
       std::string hash_string(const std::string& s);
       
       // Progress/logging (inherits notification)
       void success(std::string msg);
       void failure(std::string msg);
       void warning(std::string msg);
   };

typecasting
~~~~~~~~~~~

**Location**: ``modules/typecasting/include/tools/``

Type conversion utilities between Python/C++/ROOT.

**Key Functions**:

* ``merge_cast.h``: Merge different data types
* ``vector_cast.h``: Vector type conversions

notification
~~~~~~~~~~~~

**Header**: ``modules/notification/include/notification/notification.h``

Logging and progress notification system.

**Key Class**: ``notification``

.. code-block:: cpp

   class notification {
   public:
       notification();
       virtual ~notification();
       
       void success(std::string msg);
       void failure(std::string msg);
       void warning(std::string msg);
       void info(std::string msg);
       
       // Progress tracking
       void set_progress(double percent);
       double get_progress();
       
   protected:
       std::string format_message(std::string level, std::string msg);
   };

meta
~~~~

**Header**: ``modules/meta/include/meta/meta.h``

Metadata management for datasets.

**Key Class**: ``meta``

.. code-block:: cpp

   class meta : public tools {
   public:
       meta();
       ~meta();
       
       // Dataset information
       void set_cross_section(double xsec);
       double get_cross_section();
       
       void set_sum_of_weights(double sow);
       double get_sum_of_weights();
       
       void set_dataset_name(std::string name);
       std::string get_dataset_name();
       
       // Serialization
       void save(std::string path);
       void load(std::string path);
       
       meta_t data;
       std::string metacache_path;
   };

Build System
------------

All modules use CMake with:

.. code-block:: cmake

   add_library(module_name SHARED
       cxx/implementation.cxx
       # ... more sources
   )
   
   target_include_directories(module_name PUBLIC
       include/
   )
   
   target_link_libraries(module_name
       ROOT::Core
       ROOT::Tree
       # ... dependencies
   )

Compilation
-----------

.. code-block:: bash

   cd build
   cmake ..
   make -j$(nproc)

Or via pip:

.. code-block:: bash

   pip install -e .

Dependencies
------------

* C++17 compiler (GCC 9+, Clang 10+)
* ROOT 6.x
* HDF5 (optional, for io module)
* PyTorch C++ API (for model module)

See Also
--------

* :doc:`../technical/cpp_modules_api`: Detailed API for core modules
* :doc:`../technical/cuda_actual_api`: CUDA implementations
* :doc:`overview`: Module organization
