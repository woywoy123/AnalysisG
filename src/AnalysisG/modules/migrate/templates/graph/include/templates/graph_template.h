#ifndef GRAPH_TEMPLATE_H
#define GRAPH_TEMPLATE_H

#include <templates/particle_template.h>
#include <templates/event_template.h>
#include <templates/graph_t.h>

#include <structs/property.h>
#include <structs/event.h>

#include <torch/torch.h>

class container; 
class analysis; 
class meta; 

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

        int threadIdx = -1; 
        std::string filename = ""; 
        meta* meta_data = nullptr; 

        template <typename G>
        G* get_event(){return (G*)this -> m_event;}

        template <typename G, typename O, typename X>
        void add_graph_truth_feature(O* ev, X fx, std::string _name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "T-" + _name); 
        }


        template <typename G, typename O, typename X>
        void add_graph_data_feature(O* ev, X fx, std::string _name){
            cproperty<G, O> cdef; 
            cdef.set_getter(fx);
            cdef.set_object(ev); 
            G r = cdef; 
            this -> add_graph_feature(r, "D-" + _name); 
        }

        template <typename G, typename O, typename X>
        void add_node_truth_feature(X fx, std::string _name){
            std::vector<G> nodes_data = {}; 
            std::map<int, particle_template*>::iterator itr = this -> node_particles.begin(); 
            for (; itr != this -> node_particles.end(); ++itr){
                cproperty<G, O> cdef; 
                cdef.set_getter(fx);
                cdef.set_object((O*)itr -> second); 
                nodes_data.push_back((G)cdef); 
            }
            this -> add_node_feature(nodes_data, "T-" + _name); 
        }


        template <typename G, typename O, typename X>
        void add_node_data_feature(X fx, std::string _name){
            std::vector<G> nodes_data = {}; 

            std::map<int, particle_template*>::iterator itr = this -> node_particles.begin(); 
            for (; itr != this -> node_particles.end(); ++itr){
                cproperty<G, O> cdef; 
                cdef.set_getter(fx);
                cdef.set_object((O*)itr -> second); 
                nodes_data.push_back((G)cdef); 
            }
            this -> add_node_feature(nodes_data, "D-" + _name); 
        }

        template <typename G, typename O, typename X>
        void add_edge_truth_feature(X fx, std::string _name){
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
            this -> add_edge_feature(edge_data, "T-" + _name); 
        }


        template <typename G, typename O, typename X>
        void add_edge_data_feature(X fx, std::string _name){
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
            this -> add_edge_feature(edge_data, "D-" + _name); 
        }

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
        std::vector<particle_template*> garbage = {}; 

        std::vector<std::vector<int>> _topology; 
        std::vector<int> _topological_index;
        torch::Tensor m_topology; 

        torch::TensorOptions* op = nullptr; 
        event_template* m_event = nullptr; 

        bool m_preselection = false; 
        graph_template* build(event_template* el); 
        graph_t* data_export(); 
        event_t data; 

}; 
 


#endif
