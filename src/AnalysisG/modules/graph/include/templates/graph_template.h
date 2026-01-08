/**
 * @file graph_template.h
 * @brief Defines the graph_template class and graph_t structure for graph-based analysis.
 *
 * This file contains the declaration of the `graph_template` class and `graph_t` structure,
 * which form the foundation of graph-based machine learning analysis in AnalysisG.
 * The graph template defines how physics events are converted into graph representations,
 * while graph_t holds the actual tensor data for training and inference.
 *
 * @section graph_overview Overview
 *
 * Graph construction in AnalysisG follows this workflow:
 * 1. Define which particles become nodes via `define_particle_nodes()`
 * 2. Define edge topology via `define_topology()` 
 * 3. Add node/edge/graph features using template methods
 * 4. Export to `graph_t` for batching and GPU transfer
 *
 * @section graph_features Feature Types
 *
 * Features are prefixed to indicate their purpose:
 * - `D-*`: Data features (model inputs)
 * - `T-*`: Truth features (training targets)
 */

#ifndef GRAPH_TEMPLATE_H
#define GRAPH_TEMPLATE_H

#include <templates/particle_template.h>
#include <templates/event_template.h>

#include <structs/property.h>
#include <structs/event.h>
#include <structs/folds.h>
#include <structs/enums.h>

#include <tools/tensor_cast.h>
#include <tools/tools.h>

#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include <mutex> 

#include <pyc/pyc.h>

#ifdef PYC_CUDA
#define cu_pyc c10::kCUDA
#else
#define cu_pyc c10::kCPU
#endif


class graph_template; 
class dataloader; 
class container; 
class analysis; 
class meta; 

/**
 * @struct graph_t
 * @brief Data structure holding graph tensors for ML training and inference.
 *
 * The graph_t structure contains all tensor data for a single graph or batch of graphs.
 * It manages feature tensors, edge indices, and device placement for both CPU and CUDA.
 *
 * @section graph_t_access Accessing Features
 *
 * Use the getter methods to access features by name and device:
 * - `get_data_node()`: Get node input features
 * - `get_truth_node()`: Get node target labels
 * - `get_data_edge()`: Get edge input features
 * - `get_truth_edge()`: Get edge target labels
 * - `get_data_graph()`: Get graph-level input features
 * - `get_truth_graph()`: Get graph-level target labels
 * - `get_edge_index()`: Get edge connectivity [2, E]
 * - `get_batch_index()`: Get batch assignment per node
 *
 * @section graph_t_device Device Management
 *
 * Tensors can be transferred to CUDA devices using `transfer_to_device()`.
 * Each device maintains its own copy of the tensors.
 */
struct graph_t {

    public: 
        /**
         * @brief Get truth features for the entire graph.
         * @tparam g Model type (used to get device index).
         * @param name Feature name (without T- prefix).
         * @param mdl Pointer to model for device information.
         * @return Pointer to the tensor, or nullptr if not found.
         */
        template <typename g>
        torch::Tensor* get_truth_graph(std::string name, g* mdl){
            return this -> has_feature(graph_enum::truth_graph, name, mdl -> device_index); 
        }
        
        /**
         * @brief Get truth features for nodes.
         * @tparam g Model type.
         * @param name Feature name.
         * @param mdl Pointer to model.
         * @return Pointer to tensor [N_nodes, ...].
         */
        template <typename g>
        torch::Tensor* get_truth_node(std::string name, g* mdl){
            return this -> has_feature(graph_enum::truth_node, name, mdl -> device_index); 
        }
        
        /**
         * @brief Get truth features for edges.
         * @tparam g Model type.
         * @param name Feature name.
         * @param mdl Pointer to model.
         * @return Pointer to tensor [N_edges, ...].
         */
        template <typename g>
        torch::Tensor* get_truth_edge(std::string name, g* mdl){
            return this -> has_feature(graph_enum::truth_edge, name, mdl -> device_index); 
        }
        
        /**
         * @brief Get data features for the entire graph.
         * @tparam g Model type.
         * @param name Feature name.
         * @param mdl Pointer to model.
         * @return Pointer to tensor.
         */
        template <typename g>
        torch::Tensor* get_data_graph(std::string name, g* mdl){
            return this -> has_feature(graph_enum::data_graph, name, mdl -> device_index); 
        }
        
        /**
         * @brief Get data features for nodes.
         * @tparam g Model type.
         * @param name Feature name.
         * @param mdl Pointer to model.
         * @return Pointer to tensor [N_nodes, F].
         */
        template <typename g>
        torch::Tensor* get_data_node(std::string name, g* mdl){
            return this -> has_feature(graph_enum::data_node, name, mdl -> device_index); 
        }
        
        /**
         * @brief Get data features for edges.
         * @tparam g Model type.
         * @param name Feature name.
         * @param mdl Pointer to model.
         * @return Pointer to tensor [N_edges, F].
         */
        template <typename g>
        torch::Tensor* get_data_edge(std::string name, g* mdl){
            return this -> has_feature(graph_enum::data_edge, name, mdl -> device_index); 
        }
        
        /**
         * @brief Get edge index tensor.
         * @tparam g Model type.
         * @param mdl Pointer to model.
         * @return Pointer to tensor [2, N_edges] with source/target node indices.
         */
        template <typename g>
        torch::Tensor* get_edge_index(g* mdl){
            return this -> has_feature(graph_enum::edge_index, "", mdl -> device_index); 
        }

        /**
         * @brief Get event weight tensor.
         * @tparam g Model type.
         * @param mdl Pointer to model.
         * @return Pointer to weight tensor for loss weighting.
         */
        template <typename g>
        torch::Tensor* get_event_weight(g* mdl){
            return this -> has_feature(graph_enum::weight, "", mdl -> device_index); 
        }

        /**
         * @brief Get batch index tensor.
         * @tparam g Model type.
         * @param mdl Pointer to model.
         * @return Pointer to tensor [N_nodes] indicating which graph each node belongs to.
         */
        template <typename g>
        torch::Tensor* get_batch_index(g* mdl){
            return this -> has_feature(graph_enum::batch_index, "", mdl -> device_index); 
        }

        /**
         * @brief Get tensor of batched event indices.
         * @tparam g Model type.
         * @param mdl Pointer to model.
         * @return Pointer to tensor with original event indices.
         */
        template <typename g>
        torch::Tensor* get_batched_events(g* mdl){
            return this -> has_feature(graph_enum::batch_events, "", mdl -> device_index); 
        }

        /**
         * @brief Check if a feature exists and get its tensor.
         * @param tp Feature type enum.
         * @param name Feature name.
         * @param dev Device index.
         * @return Pointer to tensor, or nullptr if not found.
         */
        torch::Tensor* has_feature(graph_enum tp, std::string name, int dev); 
        
        void add_truth_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); ///< Add graph-level truth features.
        void add_truth_node( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); ///< Add node-level truth features.
        void add_truth_edge( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); ///< Add edge-level truth features.
        void add_data_graph( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); ///< Add graph-level data features.
        void add_data_node(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); ///< Add node-level data features.
        void add_data_edge(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); ///< Add edge-level data features.

        /**
         * @brief Transfer all tensors to a specific device.
         * @param dev TensorOptions specifying the target device.
         */
        void transfer_to_device(torch::TensorOptions* dev); 
        
        /**
         * @brief Purge all tensor data and free memory.
         */
        void _purge_all(); 

        int       num_nodes = 0;      ///< Number of nodes in the graph.
        long    event_index = 0;      ///< Source event index.
        double event_weight = 1;      ///< Event weight for training.
        bool   preselection = true;   ///< Whether graph passed preselection.
        std::vector<long> batched_events = {}; ///< Event indices when batched.

        std::string* hash       = nullptr; ///< Unique hash identifier.
        std::string* filename   = nullptr; ///< Source filename.
        std::string* graph_name = nullptr; ///< Graph template name.

        c10::DeviceType device = c10::kCPU;  ///< Current device (CPU or CUDA).
        int in_use = 1; ///< Reference count for garbage collection. 

    private:
        friend graph_template; 
        friend dataloader; 
        bool is_owner = false; 
        std::mutex mut; 

        torch::Tensor* edge_index = nullptr; 
        std::map<std::string, int>* data_map_graph = nullptr; 
        std::map<std::string, int>* data_map_node  = nullptr;         
        std::map<std::string, int>* data_map_edge  = nullptr;         

        std::map<std::string, int>* truth_map_graph = nullptr; 
        std::map<std::string, int>* truth_map_node  = nullptr;         
        std::map<std::string, int>* truth_map_edge  = nullptr;         

        std::vector<torch::Tensor*>* data_graph = nullptr; 
        std::vector<torch::Tensor*>* data_node  = nullptr; 
        std::vector<torch::Tensor*>* data_edge  = nullptr; 
          
        std::vector<torch::Tensor*>* truth_graph = nullptr; 
        std::vector<torch::Tensor*>* truth_node  = nullptr; 
        std::vector<torch::Tensor*>* truth_edge  = nullptr;

        std::map<int, std::vector<torch::Tensor>> dev_data_graph = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_data_node  = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_data_edge  = {}; 

        std::map<int, std::vector<torch::Tensor>> dev_truth_graph = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_truth_node  = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_truth_edge  = {};

        std::map<int, torch::Tensor> dev_edge_index   = {}; 
        std::map<int, torch::Tensor> dev_batch_index  = {}; 
        std::map<int, torch::Tensor> dev_event_weight = {};
        std::map<int, torch::Tensor> dev_batched_events = {};  
        std::map<int, bool> device_index = {}; 

        void meta_serialize(std::map<std::string, int>* data, std::string* out); 
        void meta_serialize(std::vector<torch::Tensor*>* data, std::string* out); 
        void meta_serialize(torch::Tensor* data, std::string* out); 
        void serialize(graph_hdf5* m_hdf5);

        void meta_deserialize(std::map<std::string, int>* data, std::string* inpt); 
        void meta_deserialize(std::vector<torch::Tensor*>* data, std::string* inpt); 
        torch::Tensor* meta_deserialize(std::string* inpt); 
        void deserialize(graph_hdf5* m_hdf5);

        void _purge_data(std::vector<torch::Tensor*>* data); 
        void _purge_data(std::map<int, torch::Tensor*>* data); 
        void _purge_data(std::map<int, std::vector<torch::Tensor*>*>* data); 
        std::vector<torch::Tensor*>* add_content(std::map<std::string, torch::Tensor*>* inpt); 

        void _transfer_to_device(
                std::vector<torch::Tensor>* trg, 
                std::vector<torch::Tensor*>* data, 
                torch::TensorOptions* dev
        ); 

        torch::Tensor* return_any(
                std::map<std::string, int>* loc, 
                std::map<int, std::vector<torch::Tensor>>* container, 
                std::string name, int dev_);
}; 


/**
 * @brief Default topology function that connects all node pairs.
 * @param p1 First particle (unused).
 * @param p2 Second particle (unused).
 * @return Always returns true (fully connected graph).
 */
bool static fulltopo(particle_template*, particle_template*){return true;}; 

/**
 * @class graph_template
 * @brief Base class for converting physics events into graph representations.
 *
 * The graph_template class provides the interface for defining how physics events
 * are converted into graphs suitable for graph neural network processing.
 *
 * @section graph_template_workflow Workflow
 *
 * To create a custom graph template:
 *
 * 1. Subclass graph_template
 * 2. Override `CompileEvent()` to define graph construction
 * 3. In `CompileEvent()`:
 *    - Call `define_particle_nodes()` to set which particles become nodes
 *    - Call `define_topology()` to set edge connectivity
 *    - Use `add_node_data_feature()` etc. to add features
 * 4. Optionally override `PreSelection()` to filter events
 *
 * @section graph_template_features Adding Features
 *
 * Features are added using template methods that accept getter functions:
 *
 * ```cpp
 * // Add node data feature (model input)
 * add_node_data_feature<double, MyParticle>([](double* out, MyParticle* p) {
 *     *out = p->pt / 1000.0;  // pT in GeV
 * }, "pt");
 *
 * // Add node truth feature (training target)
 * add_node_truth_feature<int, MyParticle>([](int* out, MyParticle* p) {
 *     *out = p->is_from_top ? 1 : 0;
 * }, "is_top");
 *
 * // Add edge feature
 * add_edge_data_feature<double, MyParticle>([](double* out, std::tuple<MyParticle*, MyParticle*>* edge) {
 *     auto [p1, p2] = *edge;
 *     *out = p1->DeltaR(p2);
 * }, "deltaR");
 * ```
 *
 * @section graph_template_topology Topology
 *
 * The topology function determines which node pairs are connected:
 *
 * ```cpp
 * define_topology([](particle_template* p1, particle_template* p2) {
 *     // Connect nodes within deltaR < 1.0
 *     return p1->DeltaR(p2) < 1.0;
 * });
 * ```
 *
 * @see graph_t
 * @see event_template
 */
class graph_template: public tools
{
    public:
        /**
         * @brief Default constructor.
         * Initializes the graph template with default settings.
         */
        graph_template(); 
        
        /**
         * @brief Virtual destructor.
         * Cleans up allocated resources.
         */
        virtual ~graph_template(); 
        
        /**
         * @brief Creates a clone of this graph template.
         * @return Pointer to the cloned instance.
         */
        virtual graph_template* clone(); 
        
        /**
         * @brief Compiles the event into a graph.
         * Override this method to define custom graph construction logic.
         * Access the event via `get_event<EventType>()`.
         */
        virtual void CompileEvent(); 
        
        /**
         * @brief Pre-selection filter for events.
         * Override to skip events that shouldn't produce graphs.
         * @return True to keep the event, false to skip.
         */
        virtual bool PreSelection();

        /**
         * @brief Defines which particles become graph nodes.
         * @param prt Vector of particles to use as nodes.
         */
        void define_particle_nodes(std::vector<particle_template*>* prt); 
        
        /**
         * @brief Defines the edge topology between nodes.
         * @param fx Function that returns true if two particles should be connected.
         */
        void define_topology(std::function<bool(particle_template*, particle_template*)> fx);

        /**
         * @brief Clears all particle references.
         */
        void flush_particles(); 
        
        /**
         * @brief Equality comparison operator.
         * @param p The graph template to compare against.
         * @return True if templates have the same hash.
         */
        bool operator == (graph_template& p); 

        cproperty<long  , graph_template> index;        ///< Event index property.
        cproperty<double, graph_template> weight;       ///< Event weight property.
        cproperty<bool  , graph_template> preselection; ///< Preselection result property.

        cproperty<std::string, graph_template> hash; ///< Unique hash identifier.
        cproperty<std::string, graph_template> tree; ///< Source tree name.
        cproperty<std::string, graph_template> name; ///< Graph template name.

        int threadIdx = -1;           ///< Thread index for parallel processing.
        std::string filename = "";    ///< Source filename.
        meta* meta_data = nullptr;    ///< Pointer to event metadata.

        /**
         * @brief Gets the source event cast to a specific type.
         * @tparam G The event type to cast to.
         * @return Pointer to the event.
         */
        template <typename G>
        G* get_event(){return (G*)this -> m_event;}

        /**
         * @brief Adds a graph-level truth feature.
         * @tparam G Feature value type.
         * @tparam O Object type to extract feature from.
         * @tparam X Getter function type.
         * @param ev Object to get feature from.
         * @param fx Getter function.
         * @param name Feature name (T- prefix added automatically).
         */
        template <typename G, typename O, typename X>
        void add_graph_truth_feature(O* ev, X fx, std::string name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "T-" + name); 
        }

        /**
         * @brief Adds a graph-level data feature.
         * @tparam G Feature value type.
         * @tparam O Object type to extract feature from.
         * @tparam X Getter function type.
         * @param ev Object to get feature from.
         * @param fx Getter function.
         * @param name Feature name (D- prefix added automatically).
         */
        template <typename G, typename O, typename X>
        void add_graph_data_feature(O* ev, X fx, std::string name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "D-" + name); 
        }

        /**
         * @brief Adds a node-level truth feature.
         * @tparam G Feature value type.
         * @tparam O Particle type.
         * @tparam X Getter function type.
         * @param fx Getter function called for each node particle.
         * @param name Feature name (T- prefix added automatically).
         */
        template <typename G, typename O, typename X>
        void add_node_truth_feature(X fx, std::string name){
            std::vector<G> nodes_data = {}; 
            std::map<int, particle_template*>::iterator itr = this -> node_particles.begin(); 
            for (; itr != this -> node_particles.end(); ++itr){
                cproperty<G, O> cdef; 
                cdef.set_getter(fx);
                cdef.set_object((O*)itr -> second); 
                nodes_data.push_back((G)cdef); 
            }
            this -> add_node_feature(nodes_data, "T-" + name); 
        }

        /**
         * @brief Adds a node-level data feature.
         * @tparam G Feature value type.
         * @tparam O Particle type.
         * @tparam X Getter function type.
         * @param fx Getter function called for each node particle.
         * @param name Feature name (D- prefix added automatically).
         */
        template <typename G, typename O, typename X>
        void add_node_data_feature(X fx, std::string name){
            std::vector<G> nodes_data = {}; 

            std::map<int, particle_template*>::iterator itr = this -> node_particles.begin(); 
            for (; itr != this -> node_particles.end(); ++itr){
                cproperty<G, O> cdef; 
                cdef.set_getter(fx);
                cdef.set_object((O*)itr -> second); 
                nodes_data.push_back((G)cdef); 
            }
            this -> add_node_feature(nodes_data, "D-" + name); 
        }

        /**
         * @brief Adds an edge-level truth feature.
         * @tparam G Feature value type.
         * @tparam O Particle type.
         * @tparam X Getter function type.
         * @param fx Getter function called for each edge (particle pair).
         * @param name Feature name (T- prefix added automatically).
         */
        template <typename G, typename O, typename X>
        void add_edge_truth_feature(X fx, std::string name){
            int dx = -1; 
            std::vector<G> edge_data = {}; 
            std::map<int, particle_template*>::iterator itr1;
            std::map<int, particle_template*>::iterator itr2;
            if (!this -> _topological_index.size()){this -> define_topology(fulltopo);} 
            for (itr1 = this -> node_particles.begin(); itr1 != this -> node_particles.end(); ++itr1){
                for (itr2 = this -> node_particles.begin(); itr2 != this -> node_particles.end(); ++itr2){
                    ++dx; 

                    if (this -> _topological_index[dx] < 0){continue;}
                    std::tuple<O*, O*> p_ij((O*)itr1 -> second, (O*)itr2 -> second); 
                    cproperty<G, std::tuple<O*, O*>> cdef; 
                    cdef.set_object(&p_ij); 
                    cdef.set_getter(fx); 
                    edge_data.push_back(cdef); 
                }
            }
            this -> add_edge_feature(edge_data, "T-" + name); 
        }

        /**
         * @brief Adds an edge-level data feature.
         * @tparam G Feature value type.
         * @tparam O Particle type.
         * @tparam X Getter function type.
         * @param fx Getter function called for each edge (particle pair).
         * @param name Feature name (D- prefix added automatically).
         */
        template <typename G, typename O, typename X>
        void add_edge_data_feature(X fx, std::string name){
            int dx = -1; 
            std::vector<G> edge_data = {}; 
            std::map<int, particle_template*>::iterator itr1;
            std::map<int, particle_template*>::iterator itr2;
            if (!this -> _topological_index.size()){this -> define_topology(fulltopo);} 
            for (itr1 = this -> node_particles.begin(); itr1 != this -> node_particles.end(); ++itr1){
                for (itr2 = this -> node_particles.begin(); itr2 != this -> node_particles.end(); ++itr2){
                    ++dx; 

                    if (this -> _topological_index[dx] < 0){continue;}
                    std::tuple<O*, O*> p_ij(itr1 -> second, itr2 -> second); 
                    cproperty<G, std::tuple<O*, O*>> cdef; 
                    cdef.set_object(&p_ij); 
                    cdef.set_getter(fx); 
                    edge_data.push_back(cdef); 
                }
            }
            this -> add_edge_feature(edge_data, "D-" + name); 
        }


        /**
         * @brief Reconstructs double neutrino momenta for di-leptonic events.
         * @param mass_top Top quark mass in MeV (default: 172620 MeV = 172.62 GeV).
         * @param mass_wboson W boson mass in MeV (default: 80385 MeV = 80.385 GeV).
         * @param top_perc Top mass constraint percentage (default 0.85).
         * @param w_perc W mass constraint percentage (default 0.95).
         * @param distance Convergence tolerance (default 1e-8).
         * @param steps Maximum iteration steps (default 10).
         * @return True if reconstruction succeeded.
         */
        bool double_neutrino(
                double mass_top = 172.62*1000, double mass_wboson = 80.385*1000, 
                double top_perc = 0.85, double w_perc = 0.95, double distance = 1e-8, int steps = 10
        ); 

    private:
        friend container; 
        friend analysis; 

        // -------- Graph Features ----------- //
        void add_graph_feature(bool, std::string);   
        void add_graph_feature(std::vector<bool>, std::string);   

        void add_graph_feature(float, std::string);   
        void add_graph_feature(std::vector<float>, std::string);   

        void add_graph_feature(double, std::string);   
        void add_graph_feature(std::vector<double>, std::string);   

        void add_graph_feature(long, std::string);   
        void add_graph_feature(std::vector<long>, std::string);   

        void add_graph_feature(int, std::string);   
        void add_graph_feature(std::vector<int>, std::string);   
        void add_graph_feature(std::vector<std::vector<int>>, std::string);   

        // -------- Node Features ----------- //
        void add_node_feature(bool, std::string);   
        void add_node_feature(std::vector<bool>, std::string);   

        void add_node_feature(float, std::string);   
        void add_node_feature(std::vector<float>, std::string);   

        void add_node_feature(double, std::string);   
        void add_node_feature(std::vector<double>, std::string);   

        void add_node_feature(long, std::string);   
        void add_node_feature(std::vector<long>, std::string);   

        void add_node_feature(int, std::string);   
        void add_node_feature(std::vector<int>, std::string);   
        void add_node_feature(std::vector<std::vector<int>>, std::string);   


        // -------- Edge Features ----------- //
        void add_edge_feature(bool, std::string);   
        void add_edge_feature(std::vector<bool>, std::string);   

        void add_edge_feature(float, std::string);   
        void add_edge_feature(std::vector<float>, std::string);   

        void add_edge_feature(double, std::string);   
        void add_edge_feature(std::vector<double>, std::string);   

        void add_edge_feature(long, std::string);   
        void add_edge_feature(std::vector<long>, std::string);   

        void add_edge_feature(int, std::string);   
        void add_edge_feature(std::vector<int>, std::string);   
        void add_edge_feature(std::vector<std::vector<int>>, std::string);   

        template <typename G, typename g>
        torch::Tensor to_tensor(std::vector<G> _data, at::ScalarType _op, g prim){
            return build_tensor(&_data, _op, prim, this -> op); 
        } 

        void static set_name(std::string*, graph_template*); 
        void static set_preselection(bool*, graph_template*); 

        void static get_hash(std::string*, graph_template*); 
        void static get_index(long*, graph_template*); 
        void static get_weight(double*, graph_template*); 
        void static get_tree(std::string*, graph_template*); 
        void static get_preselection(bool*, graph_template*); 

        void static build_export(
                std::map<std::string, torch::Tensor*>* _truth_t, std::map<std::string, int>* _truth_i,
                std::map<std::string, torch::Tensor*>* _data_t , std::map<std::string, int>*  _data_i,
                std::map<std::string, torch::Tensor>* _fx
        );


        int num_nodes = 0; 
        std::map<std::string, int> nodes = {}; 
        std::map<int, particle_template*> node_particles = {}; 
        std::map<std::string, torch::Tensor> graph_fx = {}; 
        std::map<std::string, torch::Tensor> node_fx  = {}; 
        std::map<std::string, torch::Tensor> edge_fx  = {}; 

        std::vector<std::vector<int>> _topology; 
        std::vector<int> _topological_index;
        torch::Tensor m_topology; 

        torch::TensorOptions* op = nullptr; 
        event_template* m_event = nullptr; 

        bool m_preselection = true; 
        graph_template* build(event_template* el); 
        graph_t* data_export(); 
        event_t data; 

}; 
 


#endif
