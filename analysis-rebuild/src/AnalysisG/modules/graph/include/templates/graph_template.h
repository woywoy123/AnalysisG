#ifndef GRAPH_TEMPLATE_H
#define GRAPH_TEMPLATE_H

#include <templates/particle_template.h>
#include <templates/event_template.h>
#include <structs/property.h>
#include <structs/event.h>
#include <tools/tools.h>

#include <ATen/ATen.h>
#include <torch/torch.h>

struct graph_t {
    public: 


        torch::Tensor* get_truth_graph(std::string name){
            if (!this -> truth_map_graph -> count(name)){return nullptr;}
            int x = (*this -> truth_map_graph)[name]; 
            return this -> truth_graph -> at(x); 
        }

        torch::Tensor* get_truth_node(std::string name){
            if (!this -> truth_map_node -> count(name)){return nullptr;}
            std::cout << this -> truth_map_node << std::endl;
            int x = (*this -> truth_map_node)[name]; 
            return this -> truth_node -> at(x); 
        }

        torch::Tensor* get_truth_edge(std::string name){
            if (!this -> truth_map_edge -> count(name)){return nullptr;}
            int x = (*this -> truth_map_edge)[name]; 
            return this -> truth_edge -> at(x); 
        }

        torch::Tensor* get_data_graph(std::string name){
            if (!this -> data_map_graph -> count(name)){return nullptr;}
            int x = (*this -> data_map_graph)[name]; 
            return this -> data_graph -> at(x); 
        }

        torch::Tensor* get_data_node(std::string name){
            if (!this -> data_map_node -> count(name)){return nullptr;}
            int x = (*this -> data_map_node)[name]; 
            return this -> data_node -> at(x); 
        }

        torch::Tensor* get_data_edge(std::string name){
            if (!this -> data_map_edge -> count(name)){return nullptr;}
            int x = (*this -> data_map_edge)[name]; 
            return this -> data_edge -> at(x); 
        }

        void add_truth_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
            if (this -> truth_graph){return;}
            this -> truth_graph = this -> add_content(data); 
            this -> truth_map_graph = maps; 
        }

        void add_truth_node(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
            if (this -> truth_node){return;}
            this -> truth_node = this -> add_content(data);
            this -> truth_map_node = maps; 
        }

        void add_truth_edge(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
            if (this -> truth_edge){return;}
            this -> truth_edge = this -> add_content(data);
            this -> truth_map_edge = maps; 
        }

        void add_data_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
            if (this -> data_graph){return;}
            this -> data_graph = this -> add_content(data); 
            this -> data_map_graph = maps; 
        }

        void add_data_node(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
            if (this -> data_node){return;}
            this -> data_node = this -> add_content(data);
            this -> data_map_node = maps; 
        }

        void add_data_edge(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
            if (this -> data_edge){return;}
            this -> data_edge = this -> add_content(data);
            this -> data_map_edge = maps; 
        }

        void _purge_all(){
            this -> _purge_data(this -> data_graph); 
            this -> _purge_data(this -> data_node); 
            this -> _purge_data(this -> data_edge); 
            delete this -> edge_index;

            this -> _purge_data(this -> truth_graph); 
            this -> _purge_data(this -> truth_node); 
            this -> _purge_data(this -> truth_edge); 
        }


        torch::Tensor* edge_index = nullptr; 
        long event_index = 0; 
        int num_nodes = 0; 

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

    private:
        void _purge_data(std::vector<torch::Tensor*>* data){
            if (!data){return;}
            for (size_t x(0); x < data -> size(); ++x){delete data -> at(x);}
        }

        std::vector<torch::Tensor*>* add_content(std::map<std::string, torch::Tensor*>* inpt){
            std::vector<torch::Tensor*>* out = new std::vector<torch::Tensor*>(inpt -> size(), nullptr); 
            std::map<std::string, torch::Tensor*>::iterator itr = inpt -> begin();
            for (int t(0); t < inpt -> size(); ++t, ++itr){(*out)[t] = itr -> second;}
            return out; 
        }
}; 





bool static fulltopo(particle_template*, particle_template*){return true;}; 

class graph_template: public tools
{
    private:
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


        template <typename G>
        torch::Tensor to_tensor(std::vector<G> _data, at::ScalarType _op){
            int s = _data.size(); 
            G d[s] = {0}; 
            for (int x(0); x < s; ++x){d[x] = _data.at(x);}
            torch::TensorOptions* f = this -> op; 
            return torch::from_blob(d, {s}, (*this -> op).dtype(_op)).clone(); 
        }; 

        void static set_name(std::string*, graph_template*); 
        void static get_hash(std::string*, graph_template*); 

        void static set_device(std::string*, graph_template*); 
        void static get_device(std::string*, graph_template*); 

        void static get_index(long*, graph_template*); 
        void static get_tree(std::string*, graph_template*); 

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

        std::string m_device = "";  
        torch::TensorOptions* op = nullptr; 
        event_template* m_event = nullptr; 

    public:

        graph_t* data_export(); 

        event_t data; 
        std::string filename = ""; 
        bool is_compiled = false; 

        void flush_particles(); 
        graph_template* build_event(event_template* el); 
        void define_particle_nodes(std::vector<particle_template*>* prt); 
        void define_topology(std::function<bool(particle_template*, particle_template*)> fx);

        graph_template(); 
        virtual ~graph_template(); 
        virtual graph_template* clone(); 
        virtual void CompileEvent(); 

        bool operator == (graph_template& p); 
        cproperty<long, graph_template> index; 
        cproperty<std::string, graph_template> hash; 
        cproperty<std::string, graph_template> tree;  
        cproperty<std::string, graph_template> name; 
        cproperty<std::string, graph_template> device; 


        template <typename G>
        G* get_event(){return (G*)this -> m_event;}

        template <typename G, typename O, typename X>
        void add_graph_truth_feature(O* ev, X fx, std::string name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "T-" + name); 
        };


        template <typename G, typename O, typename X>
        void add_graph_data_feature(O* ev, X fx, std::string name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "D-" + name); 
        };

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
        };


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
        };

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
        };


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
        };
}; 


#endif
