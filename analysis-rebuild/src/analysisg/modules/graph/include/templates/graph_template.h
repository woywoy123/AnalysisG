#ifndef GRAPH_TEMPLATE_H
#define GRAPH_TEMPLATE_H

#include <templates/particle_template.h>
#include <templates/event_template.h>
#include <structs/property.h>
#include <structs/event.h>
#include <tools/tools.h>

#include <ATen/ATen.h>
#include <torch/torch.h>

class graph_template: public tools
{
    public:
        graph_template(); 
        virtual ~graph_template(); 

        bool operator == (graph_template& p); 
        cproperty<long, graph_template> index; 
        cproperty<std::string, graph_template> hash; 
        cproperty<std::string, graph_template> tree;  
        cproperty<std::string, graph_template> name; 

        cproperty<std::string, graph_template> device; 
        void add_particle_nodes(std::vector<particle_template*>* prt); 

        template <typename R, typename G>
        void inv(R fx, G* ev){
            R r = fx(ev); 
        }




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

        void define_topology(std::function<bool(particle_template*, particle_template*)> fx);


        virtual graph_template* clone(); 
        virtual void build_event(event_template* el); 
        virtual void CompileEvent(); 

        event_t data; 
        std::string filename = ""; 

    private:

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

        std::map<std::string, torch::Tensor> graph_fx = {}; 
        std::map<std::string, torch::Tensor> node_fx  = {}; 
        std::map<std::string, torch::Tensor> edge_fx  = {}; 
        torch::Tensor m_topology; 

        torch::TensorOptions* op = nullptr; 

        std::map<int, particle_template*> node_particles = {}; 
        std::map<std::string, int> nodes = {}; 
        std::string m_device = "";  

        int num_nodes = 0; 
}; 


#endif
