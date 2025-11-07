====================================================================================================
Complete ROOT File I/O Documentation
====================================================================================================

This document provides **comprehensive documentation** for ROOT file reading and writing operations
in the AnalysisG framework, including complete API reference for TTree creation, branch management,
and data serialization.

.. contents::
   :local:
   :depth: 3

Overview
====================================================================================================

The ``io`` module (``modules/io/``) provides unified interfaces for:
- Reading ROOT files (TFile, TTree, TBranch navigation)
- Writing ROOT files (TTree creation, branch writing)
- HDF5 file operations (reading and writing)
- Metadata extraction and caching

**Location**: ``src/AnalysisG/modules/io/include/io/io.h``

**Dependencies**:
- ROOT framework (TFile, TTree, TBranch, TLeaf, TTreeReader)
- HDF5 C++ library (H5Cpp.h)
- meta (metadata management)
- structs (data structures)
- tools (utilities)
- notification (progress reporting)

Class Definition
====================================================================================================

.. code-block:: cpp

   class io: 
       public tools, 
       public notification
   {
       public:
           io(); 
           ~io(); 
           
           // HDF5 operations
           template <typename g>
           void write(std::vector<g>* inpt, std::string set_name);
           
           template <typename g>
           void write(g* inpt, std::string set_name);
           
           template <typename g>
           void read(std::vector<g>* outpt, std::string set_name);
           
           template <typename g>
           void read(g* out, std::string set_name);
           
           void read(graph_hdf5_w* out, std::string set_name);
           
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
           bool enable_pyami = true; 
           std::string metacache_path = "./"; 
           std::string current_working_path = "."; 
           std::map<std::string, std::string> file_paths = {};
   };

ROOT File Writing
====================================================================================================

Writing ROOT Files with TTree
----------------------------------------------------------------------------------------------------

The ``variable_t`` struct (defined in ``typecasting/include/tools/vector_cast.h``) handles 
ROOT file writing operations:

**Structure Definition**:

.. code-block:: cpp

   struct variable_t: public bsc_t 
   {
       public:
           variable_t(); 
           variable_t(bool use_external); 
           ~variable_t() override; 
           
           void create_meta(meta_t* mt);
           void build_switch(size_t s, torch::Tensor* tx); 
           void process(torch::Tensor* data, std::string* varname, TTree* tr);
           
           // Process methods for different data types
           void process(std::vector<std::vector<float>>*  data, std::string* varname, TTree* tr); 
           void process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr); 
           void process(std::vector<std::vector<long>>*   data, std::string* varname, TTree* tr); 
           void process(std::vector<std::vector<int>>*    data, std::string* varname, TTree* tr); 
           void process(std::vector<std::vector<bool>>*   data, std::string* varname, TTree* tr); 
           
           void process(std::vector<float>*  data, std::string* varname, TTree* tr); 
           void process(std::vector<double>* data, std::string* varname, TTree* tr); 
           void process(std::vector<long>*   data, std::string* varname, TTree* tr); 
           void process(std::vector<int>*    data, std::string* varname, TTree* tr); 
           void process(std::vector<bool>*   data, std::string* varname, TTree* tr); 
           
           void process(float*  data, std::string* varname, TTree* tr); 
           void process(double* data, std::string* varname, TTree* tr); 
           void process(long*   data, std::string* varname, TTree* tr); 
           void process(int*    data, std::string* varname, TTree* tr); 
           void process(bool*   data, std::string* varname, TTree* tr); 
           
           std::string variable_name = ""; 
           bool failed_branch = false; 
       
       private: 
           friend write_t;
           bool use_external = false; 
           bool is_triggered = false; 
           
           TBranch* tb = nullptr; 
           TTree*   tt = nullptr; 
           meta_t* mtx = nullptr; 
   };

Complete ROOT Writing Workflow
----------------------------------------------------------------------------------------------------

**1. Create TFile and TTree**:

.. code-block:: cpp

   #include <TFile.h>
   #include <TTree.h>
   #include <tools/vector_cast.h>
   
   // Open output ROOT file
   TFile* output_file = new TFile("output.root", "RECREATE");
   
   // Create TTree
   TTree* tree = new TTree("events", "Event data");

**2. Setup Variable Handlers**:

.. code-block:: cpp

   // Create variable handlers for each branch
   variable_t pt_var;
   variable_t eta_var;
   variable_t phi_var;
   
   // Prepare data
   std::vector<float> pt_data = {45.2, 67.8, 23.1};
   std::vector<float> eta_data = {-1.2, 0.5, 2.3};
   std::vector<float> phi_data = {1.5, -0.8, 2.9};
   
   std::string pt_name = "jet_pt";
   std::string eta_name = "jet_eta";
   std::string phi_name = "jet_phi";

**3. Process and Create Branches**:

.. code-block:: cpp

   // Process data and create branches
   pt_var.process(&pt_data, &pt_name, tree);
   eta_var.process(&eta_data, &eta_name, tree);
   phi_var.process(&phi_data, &phi_name, tree);
   
   // Check for errors
   if (pt_var.failed_branch) {
       std::cerr << "Failed to create branch: " << pt_name << std::endl;
   }

**4. Fill TTree**:

.. code-block:: cpp

   // Fill the tree (automatically done by variable_t)
   // The process() method creates branches and populates them
   tree->Fill();

**5. Write and Close**:

.. code-block:: cpp

   // Write tree to file
   tree->Write();
   
   // Close file
   output_file->Close();
   delete output_file;

Writing PyTorch Tensors to ROOT
----------------------------------------------------------------------------------------------------

The ``variable_t`` struct can convert PyTorch tensors to ROOT format:

.. code-block:: cpp

   #include <torch/torch.h>
   #include <tools/vector_cast.h>
   
   // Create tensor
   torch::Tensor data_tensor = torch::randn({100, 4});
   
   // Setup variable handler
   variable_t var;
   std::string var_name = "particle_features";
   
   // Convert tensor to ROOT format
   var.process(&data_tensor, &var_name, tree);

**Internal Process**:

1. Tensor is copied to CPU if on GPU
2. Tensor is reshaped to 1D array
3. Data is converted to appropriate C++ vector type
4. TBranch is created with correct type
5. Data is written to branch

Writing Different Data Types
----------------------------------------------------------------------------------------------------

**Scalar Values**:

.. code-block:: cpp

   float event_weight = 1.53;
   int event_number = 12345;
   bool pass_selection = true;
   
   std::string weight_name = "weight";
   std::string evtnum_name = "event_number";
   std::string pass_name = "passes";
   
   variable_t weight_var, evtnum_var, pass_var;
   weight_var.process(&event_weight, &weight_name, tree);
   evtnum_var.process(&event_number, &evtnum_name, tree);
   pass_var.process(&pass_selection, &pass_name, tree);

**1D Vectors**:

.. code-block:: cpp

   std::vector<float> jet_pts = {45.2, 67.8, 23.1, 89.4};
   std::string name = "jet_pt";
   
   variable_t var;
   var.process(&jet_pts, &name, tree);

**2D Vectors**:

.. code-block:: cpp

   std::vector<std::vector<float>> jet_constituents = {
       {1.2, 3.4, 5.6},  // Jet 1 constituents
       {2.3, 4.5},       // Jet 2 constituents
       {7.8, 9.0, 1.1, 2.2}  // Jet 3 constituents
   };
   
   std::string name = "jet_constituents";
   variable_t var;
   var.process(&jet_constituents, &name, tree);

Writing Analysis Results
----------------------------------------------------------------------------------------------------

Complete example of writing analysis output to ROOT file:

.. code-block:: cpp

   #include <TFile.h>
   #include <TTree.h>
   #include <tools/vector_cast.h>
   
   void write_analysis_output(std::string output_path) {
       // Create output file
       TFile* file = new TFile(output_path.c_str(), "RECREATE");
       TTree* tree = new TTree("analysis", "Analysis results");
       
       // Event-level variables
       variable_t event_number_var, weight_var, njets_var;
       int event_number = 0;
       float weight = 1.0;
       int njets = 0;
       
       std::string evtnum_name = "event_number";
       std::string weight_name = "weight";
       std::string njets_name = "n_jets";
       
       event_number_var.process(&event_number, &evtnum_name, tree);
       weight_var.process(&weight, &weight_name, tree);
       njets_var.process(&njets, &njets_name, tree);
       
       // Jet-level variables
       variable_t jet_pt_var, jet_eta_var, jet_phi_var, jet_btag_var;
       std::vector<float> jet_pt, jet_eta, jet_phi;
       std::vector<bool> jet_btag;
       
       std::string pt_name = "jet_pt";
       std::string eta_name = "jet_eta";
       std::string phi_name = "jet_phi";
       std::string btag_name = "jet_is_b";
       
       jet_pt_var.process(&jet_pt, &pt_name, tree);
       jet_eta_var.process(&jet_eta, &eta_name, tree);
       jet_phi_var.process(&jet_phi, &phi_name, tree);
       jet_btag_var.process(&jet_btag, &btag_name, tree);
       
       // Loop over events
       for (int evt = 0; evt < num_events; ++evt) {
           event_number = evt;
           weight = get_event_weight(evt);
           
           // Get jet data
           jet_pt = get_jet_pts(evt);
           jet_eta = get_jet_etas(evt);
           jet_phi = get_jet_phis(evt);
           jet_btag = get_jet_btags(evt);
           njets = jet_pt.size();
           
           // Update variables
           event_number_var.process(&event_number, &evtnum_name, tree);
           weight_var.process(&weight, &weight_name, tree);
           njets_var.process(&njets, &njets_name, tree);
           jet_pt_var.process(&jet_pt, &pt_name, tree);
           jet_eta_var.process(&jet_eta, &eta_name, tree);
           jet_phi_var.process(&jet_phi, &phi_name, tree);
           jet_btag_var.process(&jet_btag, &btag_name, tree);
           
           // Fill tree
           tree->Fill();
       }
       
       // Write and close
       tree->Write();
       file->Close();
       delete file;
   }

ROOT File Reading
====================================================================================================

Reading ROOT Files with io Class
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   #include <io/io.h>
   
   // Create io instance
   io reader;
   
   // Set file paths
   reader.file_paths["ttbar"] = "/path/to/ttbar.root";
   reader.file_paths["signal"] = "/path/to/signal.root";
   
   // Check paths exist
   reader.check_root_file_paths();
   
   // Scan file structure
   if (!reader.scan_keys()) {
       std::cerr << "Failed to scan ROOT keys" << std::endl;
       return;
   }
   
   // Get tree sizes
   std::map<std::string, long> sizes = reader.root_size();
   for (auto& [name, size] : sizes) {
       std::cout << name << ": " << size << " entries" << std::endl;
   }

Accessing ROOT Data
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   // Begin ROOT reading
   reader.root_begin();
   
   // Get data structures
   std::map<std::string, data_t*>* data = reader.get_data();
   
   // Access specific tree
   data_t* ttbar_data = (*data)["ttbar"];
   
   // Iterate through events
   for (long entry = 0; entry < ttbar_data->entries; ++entry) {
       // Read event data
       // Process branches, leaves, etc.
   }
   
   // End reading
   reader.root_end();

ROOT Metadata Extraction
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   // Import analysis settings
   settings_t settings;
   settings.output_path = "./output/";
   reader.import_settings(&settings);
   
   // Extract metadata
   std::map<std::string, meta*> metadata = analysis_obj.meta_data;
   
   // Access sample metadata
   meta* sample_meta = metadata["ttbar"];
   if (sample_meta) {
       std::cout << "Dataset: " << sample_meta->dataset << std::endl;
       std::cout << "Sample: " << sample_meta->sample << std::endl;
       std::cout << "Is MC: " << sample_meta->isMC << std::endl;
   }

HDF5 File Operations
====================================================================================================

Writing HDF5 Files
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   #include <io/io.h>
   
   // Create io instance
   io writer;
   
   // Open HDF5 file for writing
   if (!writer.start("output.h5", "write")) {
       std::cerr << "Failed to open HDF5 file" << std::endl;
       return;
   }
   
   // Write vector data
   std::vector<float> data = {1.2, 3.4, 5.6, 7.8};
   writer.write(&data, "dataset_name");
   
   // Write scalar
   float scalar = 3.14;
   writer.write(&scalar, "pi_value");
   
   // Close file
   writer.end();

Reading HDF5 Files
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   // Open HDF5 file for reading
   io reader;
   if (!reader.start("input.h5", "read")) {
       std::cerr << "Failed to open HDF5 file" << std::endl;
       return;
   }
   
   // List datasets
   std::vector<std::string> datasets = reader.dataset_names();
   for (const std::string& name : datasets) {
       std::cout << "Dataset: " << name << std::endl;
   }
   
   // Read vector data
   std::vector<float> data;
   reader.read(&data, "dataset_name");
   
   // Read scalar
   float value;
   reader.read(&value, "pi_value");
   
   // Close file
   reader.end();

Integration with Analysis Pipeline
====================================================================================================

The ``analysis`` class uses the ``io`` module internally:

.. code-block:: cpp

   // In analysis.cxx
   static int add_content(
       std::map<std::string, torch::Tensor*>* data, 
       std::vector<variable_t>* content, 
       int index, 
       std::string prefx, 
       TTree* tt = nullptr
   ) {
       // Iterate through tensor data
       for (auto& [key, tensor] : *data) {
           variable_t var;
           std::string var_name = prefx + "_" + key;
           
           // Process tensor and create ROOT branch
           var.process(tensor, &var_name, tt);
           
           // Store variable for later use
           content->push_back(var);
       }
       return 0;
   }

This is called during inference to write model outputs to ROOT files.

Best Practices
====================================================================================================

**1. Always Check Branch Creation**:

.. code-block:: cpp

   variable_t var;
   var.process(&data, &name, tree);
   if (var.failed_branch) {
       // Handle error
   }

**2. Use Appropriate Data Types**:

- Use ``float`` for kinematics (pt, eta, phi, mass)
- Use ``int`` for counts, indices
- Use ``bool`` for flags
- Use ``long`` for large integers (run/event numbers)

**3. Organize Variables Logically**:

.. code-block:: cpp

   // Event-level first
   event_number, weight, n_jets, n_leptons
   
   // Object-level second
   jet_pt[], jet_eta[], jet_phi[]
   lepton_pt[], lepton_eta[], lepton_phi[]

**4. Memory Management**:

.. code-block:: cpp

   // Variable lifetime should match TTree lifetime
   // Keep variable_t objects alive while TTree exists
   
   std::vector<variable_t> variables;
   // ... use variables ...
   tree->Fill();
   // variables automatically cleaned up when going out of scope

**5. Batch Writing**:

.. code-block:: cpp

   // For large datasets, write in batches
   const int BATCH_SIZE = 10000;
   for (int batch = 0; batch < num_batches; ++batch) {
       // Process batch
       for (int i = 0; i < BATCH_SIZE; ++i) {
           // Fill variables
           tree->Fill();
       }
       // Periodically flush to disk
       if (batch % 10 == 0) {
           tree->AutoSave("SaveSelf");
       }
   }

