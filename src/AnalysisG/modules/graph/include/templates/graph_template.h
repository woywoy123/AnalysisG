#ifndef GRAPH_TEMPLATE_H
#define GRAPH_TEMPLATE_H

#include <templates/particle_template.h>
#include <templates/event_template.h>

#include <structs/property.h>
#include <structs/event.h>
#include <structs/folds.h>

#include <tools/tensor_cast.h>
#include <tools/tools.h>

#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include <mutex> 

#ifdef PYC_CUDA
#include <pyc/cupyc.h>
#define cu_pyc c10::kCUDA
#else
#include <pyc/tpyc.h>
#define cu_pyc c10::kCPU
#endif


class graph_template; 
class dataloader; 
class container; 
class analysis; 
class meta; 

struct graph_t {

    public: 
        template <typename g>
        torch::Tensor* get_truth_graph(std::string name, g* mdl){
            return this -> return_any(this -> truth_map_graph, &this -> dev_truth_graph, "T-" + name, mdl); 
        }
        
        template <typename g>
        torch::Tensor* get_truth_node(std::string name, g* mdl){
            return this -> return_any(this -> truth_map_node, &this -> dev_truth_node, "T-" + name, mdl); 
        }
        
        template <typename g>
        torch::Tensor* get_truth_edge(std::string name, g* mdl){
            return this -> return_any(this -> truth_map_edge, &this -> dev_truth_edge, "T-" + name, mdl); 
        }
        
        template <typename g>
        torch::Tensor* get_data_graph(std::string name, g* mdl){
            return this -> return_any(this -> data_map_graph, &this -> dev_data_graph, "D-" + name, mdl); 
        }
        
        template <typename g>
        torch::Tensor* get_data_node(std::string name, g* mdl){
            return this -> return_any(this -> data_map_node, &this -> dev_data_node, "D-" + name, mdl); 
        }
        
        template <typename g>
        torch::Tensor* get_data_edge(std::string name, g* mdl){
            return this -> return_any(this -> data_map_edge, &this -> dev_data_edge, "D-" + name, mdl); 
        }
        
        template <typename g>
        torch::Tensor* get_edge_index(g* mdl){
            return &this -> dev_edge_index[(int)mdl -> m_option -> device().index()]; 
        }

        template <typename g>
        torch::Tensor* get_event_weight(g* mdl){
            return &this -> dev_event_weight[(int)mdl -> m_option -> device().index()]; 
        }

        template <typename g>
        torch::Tensor* get_batch_index(g* mdl){
            return &this -> dev_batch_index[(int)mdl -> m_option -> device().index()]; 
        }

        template <typename g>
        torch::Tensor* get_batched_events(g* mdl){
            return &this -> dev_batched_events[(int)mdl -> m_option -> device().index()]; 
        }

        void add_truth_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_truth_node( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_truth_edge( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_data_graph( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_data_node(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_data_edge(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 

        void transfer_to_device(torch::TensorOptions* dev); 
        void _purge_all(); 

        int       num_nodes = 0; 
        long    event_index = 0; 
        double event_weight = 1; 
        bool   preselection = true;
        std::vector<long> batched_events = {}; 

        std::string* hash       = nullptr; 
        std::string* filename   = nullptr; 
        std::string* graph_name = nullptr; 

        c10::DeviceType device = c10::kCPU;  
        int in_use = 1; 

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

        template <typename g>
        torch::Tensor* return_any(
                std::map<std::string, int>* loc, 
                std::map<int, std::vector<torch::Tensor>>* container, 
                std::string name, g* mdl
        ){
            if (!loc -> count(name)){return nullptr;}
            int dev_ = (int)mdl -> m_option -> device().index(); 
            int x = (*loc)[name]; 
            return &(*container)[dev_][x];
        }

}; 


bool static fulltopo(particle_template*, particle_template*){return true;}; 

class graph_template: public tools
{
    public:
        graph_template(); 
        virtual ~graph_template(); 
        virtual graph_template* clone(); 
        virtual void CompileEvent(); 
        virtual bool PreSelection();

        void define_particle_nodes(std::vector<particle_template*>* prt); 
        void define_topology(std::function<bool(particle_template*, particle_template*)> fx);

        void flush_particles(); 
        bool operator == (graph_template& p); 

        cproperty<long  , graph_template> index; 
        cproperty<double, graph_template> weight; 
        cproperty<bool  , graph_template> preselection; 

        cproperty<std::string, graph_template> hash; 
        cproperty<std::string, graph_template> tree;  
        cproperty<std::string, graph_template> name; 

        std::string filename = ""; 
        meta* meta_data = nullptr; 

        template <typename G>
        G* get_event(){return (G*)this -> m_event;}

        template <typename G, typename O, typename X>
        void add_graph_truth_feature(O* ev, X fx, std::string name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "T-" + name); 
        }


        template <typename G, typename O, typename X>
        void add_graph_data_feature(O* ev, X fx, std::string name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "D-" + name); 
        }

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
