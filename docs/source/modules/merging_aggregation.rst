====================================================================================================
Complete Merging and Aggregation Documentation
====================================================================================================

This document provides **comprehensive documentation** for data merging, aggregation, and combination
operations in the AnalysisG framework, including template functions, selection merging, and
multi-sample aggregation strategies.

.. contents::
   :local:
   :depth: 3

Overview
====================================================================================================

The merging and aggregation system provides:
- Template functions for combining data structures
- Selection template merging across samples
- Event and graph aggregation
- Hierarchical data combination strategies

**Primary Module**: ``typecasting`` (``modules/typecasting/include/tools/merge_cast.h``)

**Related Modules**:
- ``selection`` - Selection template merging (``selection_template::merge()``, ``selection_template::merger()``)
- ``container`` - Multi-sample data management
- ``analysis`` - Orchestration of merging operations

Template Merge Functions
====================================================================================================

Core Merging Templates
----------------------------------------------------------------------------------------------------

The ``merge_cast.h`` header provides generic template functions for merging different data types:

**Location**: ``src/AnalysisG/modules/typecasting/include/tools/merge_cast.h``

**1. merge_data() - Combine Data Structures**

Merges data from one container into another:

.. code-block:: cpp

   // Vector merging - concatenates vectors
   template <typename G>
   void merge_data(std::vector<G>* out, std::vector<G>* p2) {
       out->insert(out->end(), p2->begin(), p2->end()); 
   }
   
   // Scalar merging - overwrites
   template <typename G>
   void merge_data(G* out, G* p2) {
       (*out) = *p2;
   }
   
   // Map merging - recursive merge of values
   template <typename g, typename G>
   void merge_data(std::map<g, G>* out, std::map<g, G>* p2) {
       typename std::map<g, G>::iterator itr = p2->begin(); 
       for (; itr != p2->end(); ++itr) {
           merge_data(&(*out)[itr->first], &itr->second);
       } 
   }

**Use Cases**:
- Combining selections across multiple samples
- Aggregating event data from different sources
- Merging histogram results

**Example - Merging Vectors**:

.. code-block:: cpp

   #include <tools/merge_cast.h>
   
   std::vector<float> sample1_jets = {45.2, 67.8, 23.1};
   std::vector<float> sample2_jets = {89.4, 12.7};
   
   // Merge sample2 into sample1
   merge_data(&sample1_jets, &sample2_jets);
   // Result: sample1_jets = {45.2, 67.8, 23.1, 89.4, 12.7}

**Example - Merging Maps**:

.. code-block:: cpp

   std::map<std::string, std::vector<float>> sample1_data;
   sample1_data["jet_pt"] = {45.2, 67.8};
   sample1_data["jet_eta"] = {-1.2, 0.5};
   
   std::map<std::string, std::vector<float>> sample2_data;
   sample2_data["jet_pt"] = {23.1};
   sample2_data["jet_eta"] = {2.3};
   
   // Recursive merge
   merge_data(&sample1_data, &sample2_data);
   // Result:
   // sample1_data["jet_pt"] = {45.2, 67.8, 23.1}
   // sample1_data["jet_eta"] = {-1.2, 0.5, 2.3}

**2. sum_data() - Accumulate Data**

Accumulates data with addition operation:

.. code-block:: cpp

   // Scalar summation
   template <typename G>
   void sum_data(G* out, G* p2) {
       (*out) += (*p2);
   }
   
   // Vector concatenation (same as merge_data for vectors)
   template <typename G>
   void sum_data(std::vector<G>* out, std::vector<G>* p2) {
       out->insert(out->end(), p2->begin(), p2->end()); 
   }
   
   // Map recursive summation
   template <typename g, typename G>
   void sum_data(std::map<g, G>* out, std::map<g, G>* p2) {
       typename std::map<g, G>::iterator itr = p2->begin(); 
       for (; itr != p2->end(); ++itr) {
           sum_data(&(*out)[itr->first], &itr->second);
       } 
   }

**Use Cases**:
- Accumulating event counts
- Summing weights across samples
- Combining histograms

**Example - Sum of Weights**:

.. code-block:: cpp

   float total_weight = 1523.4;
   float sample_weight = 876.2;
   
   sum_data(&total_weight, &sample_weight);
   // Result: total_weight = 2399.6

**Example - Accumulating Histograms**:

.. code-block:: cpp

   std::map<int, int> hist1;  // Bin -> Count
   hist1[0] = 120;
   hist1[1] = 450;
   hist1[2] = 230;
   
   std::map<int, int> hist2;
   hist2[0] = 80;
   hist2[1] = 310;
   hist2[2] = 190;
   
   sum_data(&hist1, &hist2);
   // Result: hist1[0] = 200, hist1[1] = 760, hist1[2] = 420

**3. contract_data() - Flatten Nested Structures**

Converts nested structures into flat vectors:

.. code-block:: cpp

   // Add single element
   template <typename g>
   void contract_data(std::vector<g>* out, g* p2) {
       out->push_back(*p2);
   }
   
   // Flatten 1D vector
   template <typename g>
   void contract_data(std::vector<g>* out, std::vector<g>* p2) {
       for (size_t i(0); i < p2->size(); ++i) {
           contract_data(out, &p2->at(i));
       }
   }
   
   // Flatten 2D vector with reservation
   template <typename g>
   void contract_data(std::vector<g>* out, std::vector<std::vector<g>>* p2) {
       long ix = 0;
       reserve_count(p2, &ix);
       out->reserve(ix); 
       for (size_t i(0); i < p2->size(); ++i) {
           contract_data(out, &p2->at(i));
       }
   }

**Use Cases**:
- Converting event-wise data to flat arrays
- Preparing data for machine learning
- Flattening jet collections across events

**Example - Flatten Jet Collections**:

.. code-block:: cpp

   // Per-event jet pT collections
   std::vector<std::vector<float>> event_jets = {
       {45.2, 67.8, 23.1},   // Event 1: 3 jets
       {89.4, 12.7},         // Event 2: 2 jets
       {34.5, 56.7, 78.9, 90.1}  // Event 3: 4 jets
   };
   
   // Flatten to single vector
   std::vector<float> all_jets;
   contract_data(&all_jets, &event_jets);
   // Result: all_jets = {45.2, 67.8, 23.1, 89.4, 12.7, 34.5, 56.7, 78.9, 90.1}

**4. reserve_count() - Pre-calculate Size**

Recursively counts elements for vector reservation:

.. code-block:: cpp

   template <typename g>
   void reserve_count(g* inp, long* ix) {
       *ix += 1;
   }
   
   template <typename g>
   void reserve_count(std::vector<g>* inp, long* ix) {
       for (size_t x(0); x < inp->size(); ++x) {
           reserve_count(&inp->at(x), ix);
       }
   }

**Use Cases**:
- Optimizing memory allocation
- Pre-calculating total element count

**Example**:

.. code-block:: cpp

   std::vector<std::vector<int>> nested = {{1, 2}, {3}, {4, 5, 6}};
   long count = 0;
   reserve_count(&nested, &count);
   // Result: count = 6

Selection Template Merging
====================================================================================================

Overview
----------------------------------------------------------------------------------------------------

The ``selection_template`` class provides two merging methods:
1. ``merge()`` - User-overridable method for custom merging logic
2. ``merger()`` - Internal method that calls ``merge()`` and handles bookkeeping

**Location**: ``src/AnalysisG/modules/selection/include/templates/selection_template.h``

merge() Method - User Interface
----------------------------------------------------------------------------------------------------

**Signature**:

.. code-block:: cpp

   virtual void merge(selection_template* sel);

**Purpose**: User-defined logic for merging selection results from another selection instance

**Override Pattern**:

.. code-block:: cpp

   class MySelection : public selection_template {
       public:
           // Custom merging logic
           void merge(selection_template* other) override {
               MySelection* other_sel = dynamic_cast<MySelection*>(other);
               if (!other_sel) return;
               
               // Merge your custom data
               merge_data(&this->my_jets, &other_sel->my_jets);
               merge_data(&this->my_leptons, &other_sel->my_leptons);
               sum_data(&this->total_weight, &other_sel->total_weight);
           }
           
           std::vector<Jet*> my_jets;
           std::vector<Lepton*> my_leptons;
           float total_weight = 0.0;
   };

**When Called**: Automatically invoked by the analysis framework when combining selections across samples

merger() Method - Internal Framework
----------------------------------------------------------------------------------------------------

**Signature**:

.. code-block:: cpp

   void merger(selection_template* sl2);

**Purpose**: Internal method that:
1. Calls user's ``merge()`` method
2. Handles internal bookkeeping
3. Manages sequence tracking
4. Coordinates with write operations

**Not User-Overridable**: This method handles framework internals

Complete Merging Workflow
====================================================================================================

Multi-Sample Analysis Merging
----------------------------------------------------------------------------------------------------

When analyzing multiple samples, the framework merges results:

**1. Per-Sample Processing**:

.. code-block:: cpp

   // Sample 1: ttbar
   MySelection* ttbar_sel = new MySelection();
   ttbar_sel->process_sample("ttbar.root");
   
   // Sample 2: signal
   MySelection* signal_sel = new MySelection();
   signal_sel->process_sample("signal.root");

**2. Automatic Merging**:

.. code-block:: cpp

   // Framework internally calls:
   ttbar_sel->merger(signal_sel);
   // Which calls:
   ttbar_sel->merge(signal_sel);  // User-defined logic

**3. Result Combination**:

.. code-block:: cpp

   // After merge, ttbar_sel contains combined results:
   // - All jets from both samples
   // - All leptons from both samples
   // - Sum of weights from both samples

Example: Multi-Sample Selection Merging
----------------------------------------------------------------------------------------------------

Complete example showing selection merging across samples:

.. code-block:: cpp

   #include <templates/selection_template.h>
   #include <tools/merge_cast.h>
   
   class TopAnalysisSelection : public selection_template {
       public:
           void merge(selection_template* other) override {
               auto* other_top = dynamic_cast<TopAnalysisSelection*>(other);
               if (!other_top) return;
               
               // Merge event counts
               sum_data(&this->n_events_processed, &other_top->n_events_processed);
               sum_data(&this->n_events_passed, &other_top->n_events_passed);
               
               // Merge selected objects
               merge_data(&this->selected_tops, &other_top->selected_tops);
               merge_data(&this->selected_jets, &other_top->selected_jets);
               merge_data(&this->selected_leptons, &other_top->selected_leptons);
               
               // Merge histograms (maps)
               merge_data(&this->top_mass_hist, &other_top->top_mass_hist);
               merge_data(&this->jet_pt_hist, &other_top->jet_pt_hist);
               
               // Sum weights
               sum_data(&this->total_weight, &other_top->total_weight);
           }
           
           // Event statistics
           long n_events_processed = 0;
           long n_events_passed = 0;
           
           // Selected objects
           std::vector<Top*> selected_tops;
           std::vector<Jet*> selected_jets;
           std::vector<Lepton*> selected_leptons;
           
           // Histograms
           std::map<int, float> top_mass_hist;  // Bin -> Weight
           std::map<int, float> jet_pt_hist;
           
           // Weights
           float total_weight = 0.0;
   };
   
   // Usage in analysis
   void run_multi_sample_analysis() {
       TopAnalysisSelection* combined = new TopAnalysisSelection();
       
       // Process ttbar sample
       TopAnalysisSelection* ttbar = new TopAnalysisSelection();
       // ... process ttbar events ...
       ttbar->n_events_processed = 100000;
       ttbar->n_events_passed = 5230;
       ttbar->total_weight = 15432.5;
       
       // Merge ttbar into combined
       combined->merger(ttbar);
       
       // Process signal sample
       TopAnalysisSelection* signal = new TopAnalysisSelection();
       // ... process signal events ...
       signal->n_events_processed = 50000;
       signal->n_events_passed = 1245;
       signal->total_weight = 8234.7;
       
       // Merge signal into combined
       combined->merger(signal);
       
       // Result: combined now contains:
       // n_events_processed = 150000
       // n_events_passed = 6475
       // total_weight = 23667.2
       // All selected objects from both samples
   }

Advanced Merging Patterns
====================================================================================================

Conditional Merging
----------------------------------------------------------------------------------------------------

Only merge if certain conditions are met:

.. code-block:: cpp

   void merge(selection_template* other) override {
       auto* other_sel = dynamic_cast<MySelection*>(other);
       if (!other_sel) return;
       
       // Only merge if selections are compatible
       if (this->analysis_mode != other_sel->analysis_mode) {
           return;  // Different modes, don't merge
       }
       
       // Conditional data merging
       if (this->include_systematics && other_sel->include_systematics) {
           merge_data(&this->systematic_variations, 
                     &other_sel->systematic_variations);
       }
       
       // Always merge baseline results
       merge_data(&this->baseline_results, &other_sel->baseline_results);
   }

Weighted Merging
----------------------------------------------------------------------------------------------------

Merge with sample-specific weighting:

.. code-block:: cpp

   void merge(selection_template* other) override {
       auto* other_sel = dynamic_cast<MySelection*>(other);
       if (!other_sel) return;
       
       // Get relative weights
       float this_weight = this->total_weight;
       float other_weight = other_sel->total_weight;
       float total = this_weight + other_weight;
       
       // Weighted average for central values
       this->avg_jet_pt = (this->avg_jet_pt * this_weight + 
                          other_sel->avg_jet_pt * other_weight) / total;
       
       // Sum for counts and total weight
       sum_data(&this->total_weight, &other_sel->total_weight);
       merge_data(&this->all_jets, &other_sel->all_jets);
   }

Hierarchical Merging
----------------------------------------------------------------------------------------------------

Merge nested data structures:

.. code-block:: cpp

   void merge(selection_template* other) override {
       auto* other_sel = dynamic_cast<MySelection*>(other);
       if (!other_sel) return;
       
       // Merge maps of vectors
       merge_data(&this->category_events, &other_sel->category_events);
       // Result: Each category's event list is concatenated
       
       // Merge maps of maps
       merge_data(&this->region_histograms, &other_sel->region_histograms);
       // Result: Each region's histograms are recursively merged
   }

Integration with Analysis Pipeline
====================================================================================================

The ``analysis`` class orchestrates merging operations:

.. code-block:: cpp

   // In analysis::build_selections()
   void build_selections() {
       selection_template* combined = nullptr;
       
       // Process each sample
       for (auto& [sample_name, sample_path] : file_labels) {
           // Create selection for this sample
           selection_template* sel = selection_names[sample_name]->clone();
           
           // Process sample events
           process_sample(sel, sample_path);
           
           // Merge into combined results
           if (!combined) {
               combined = sel;
           } else {
               combined->merger(sel);
               delete sel;
           }
       }
       
       // combined now contains results from all samples
   }

Best Practices
====================================================================================================

**1. Always Use merge_data() for Standard Types**:

.. code-block:: cpp

   // Good
   merge_data(&this->jets, &other->jets);
   
   // Bad (manual iteration)
   for (auto& jet : other->jets) {
       this->jets.push_back(jet);
   }

**2. Use sum_data() for Accumulation**:

.. code-block:: cpp

   // Event counts, weights, statistics
   sum_data(&this->n_events, &other->n_events);
   sum_data(&this->total_weight, &other->total_weight);

**3. Dynamic Cast for Safety**:

.. code-block:: cpp

   void merge(selection_template* other) override {
       auto* typed_other = dynamic_cast<MySelection*>(other);
       if (!typed_other) {
           std::cerr << "Type mismatch in merge!" << std::endl;
           return;
       }
       // Safe to use typed_other now
   }

**4. Document Merge Behavior**:

.. code-block:: cpp

   /**
    * Merges another TopSelection into this one.
    * 
    * Merge behavior:
    * - Event counts: summed
    * - Selected objects: concatenated
    * - Histograms: bin-wise summed
    * - Weights: summed
    */
   void merge(selection_template* other) override;

**5. Test Merging Logic**:

.. code-block:: cpp

   void test_merge() {
       MySelection sel1, sel2;
       
       // Setup test data
       sel1.jets = {jet1, jet2};
       sel1.total_weight = 100.0;
       
       sel2.jets = {jet3};
       sel2.total_weight = 50.0;
       
       // Merge
       sel1.merge(&sel2);
       
       // Verify
       assert(sel1.jets.size() == 3);
       assert(sel1.total_weight == 150.0);
   }

